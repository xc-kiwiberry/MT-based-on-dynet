/**
 * \file encdec.h
 * \defgroup seq2seqbuilders seq2seqbuilders
 * \brief Sequence to sequence models
 * 
 * An example implementation of a simple sequence to sequence model based on lstm encoder/decoder
 *
 */

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/tensor.h"
#include "dynet/gru.h"

#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <utility>
#include <unistd.h>
#include <float.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

//int kSOS;
int kEOS;


unsigned SRC_VOCAB_SIZE; // depends on dict
unsigned TGT_VOCAB_SIZE; // depends on dict


template <class Builder>
struct EncoderDecoder {
private:
    // Hyperparameters
    unsigned LAYERS;
    unsigned INPUT_DIM;
    unsigned HIDDEN_DIM;
    unsigned ATTENTION_SIZE;

    Builder dec_builder;     // GRU
    Builder fwd_enc_builder;
    Builder bwd_enc_builder;

    LookupParameter p_c;
    LookupParameter p_ec;  // map input to embedding (used in fwd and rev models)

    Parameter p_hid2hid;
    Parameter p_b_hid;

    Parameter attention_w1;
    Parameter attention_w2;
    Parameter attention_v;

    Parameter p_readout_allthree;
    //Parameter p_readout_hidden;
    Parameter p_readout_offset;

    Parameter p_hid2emb;
    Parameter p_emb2voc;
    Parameter p_b_voc;
public:
    /**
     * \brief Default builder
     */
    EncoderDecoder() {}

    /**
     * \brief Creates an EncoderDecoder
     *
     * \param model Model holding the parameters
     * \param num_layers Number of layers (same in the ecoder and decoder)
     * \param input_dim Dimension of the word/char embeddings
     * \param hidden_dim Dimension of the hidden states
     * \param bwd Set to `true` to make the encoder bidirectional. This doubles the number
     * of parameters in the encoder. This will also add parameters for an affine transformation
     * from the bidirectional encodings (of size num_layers * 2 * hidden_dim) to encodings
     * of size num_layers * hidden_dim compatible with the decoder
     *
     */
    explicit EncoderDecoder(Model& model,
                            unsigned num_layers,
                            unsigned input_dim,
                            unsigned hidden_dim,
                            unsigned attention_size) :
        LAYERS(num_layers), INPUT_DIM(input_dim), HIDDEN_DIM(hidden_dim), ATTENTION_SIZE(attention_size),
        dec_builder(num_layers, input_dim + hidden_dim * 2, hidden_dim, model),
        fwd_enc_builder(num_layers, input_dim, hidden_dim, model),
        bwd_enc_builder(num_layers, input_dim, hidden_dim, model) {

        p_ec = model.add_lookup_parameters(SRC_VOCAB_SIZE, {INPUT_DIM});
        p_c = model.add_lookup_parameters(TGT_VOCAB_SIZE, {INPUT_DIM});

        p_hid2hid = model.add_parameters( { HIDDEN_DIM, HIDDEN_DIM });
        p_b_hid = model.add_parameters( { HIDDEN_DIM });

        attention_w1 = model.add_parameters( { ATTENTION_SIZE, HIDDEN_DIM * 2 });
        attention_w2 = model.add_parameters( { ATTENTION_SIZE, HIDDEN_DIM * LAYERS }); // GRU
        //attention_w2 = model.add_parameters( { ATTENTION_SIZE, HIDDEN_DIM * LAYERS * 2 }); // LSTM
        attention_v = model.add_parameters( { 1, ATTENTION_SIZE });

        p_readout_allthree = model.add_parameters( { HIDDEN_DIM, INPUT_DIM + HIDDEN_DIM * 2 + HIDDEN_DIM });
        //p_readout_hidden = model.add_parameters( { HIDDEN_DIM, HIDDEN_DIM });
        p_readout_offset = model.add_parameters( { HIDDEN_DIM });

        p_hid2emb = model.add_parameters( { INPUT_DIM, HIDDEN_DIM } );
        p_emb2voc = model.add_parameters( { TGT_VOCAB_SIZE, INPUT_DIM } );
        p_b_voc = model.add_parameters( { TGT_VOCAB_SIZE } );
    }

    /**
     * \brief Batched encoding
     * \details Encodes a batch of sentences of the same size (don't forget to pad them)
     *
     * \param isents Whole dataset
     * \param id Index of the start of the batch
     * \param bsize Batch size
     * \param chars Number of tokens processed (used to compute loss per characters)
     * \param cg Computation graph
     * \return Returns the expression for the negative (batched) encoding
     */
    vector<Expression> encode(const vector<vector<int>>& isents,
                      const vector<vector<float>>& x_mask,
                      unsigned id,
                      unsigned bsize,
                      unsigned & chars,
                      ComputationGraph & cg) {
        // Set variables for the input sentence
        const unsigned islen = isents[id].size();
        vector<unsigned> x_t(bsize);
        vector<float> x_m(bsize);
        vector<Expression> i_x_t(islen);
        vector<Expression> mask(islen);
        for (int t = 0; t < islen; ++t) {
            for (int i = 0; i < bsize; ++i) {
                x_t[i] = isents[id + i][t];
                x_m[i] = x_mask[id + i][t];
            }
            i_x_t[t] = lookup(cg, p_ec, x_t);
            mask[t] = input(cg, Dim({1}, bsize), x_m);
        }

        // Forward encoder -------------------------------------------------------------------------

        // Initialize parameters in fwd_enc_builder
        fwd_enc_builder.new_graph(cg);
        // Initialize the sequence
        fwd_enc_builder.start_new_sequence();

        vector<Expression> fwd_vectors; // fwd_vectors[i] -> (hidden_dim,1)
        //Expression last_h = input(cg, Dim({HIDDEN_DIM}, bsize), vector<float>(HIDDEN_DIM * bsize));
        // Run the forward encoder on the batch
        for (int t = 0; t < islen; ++t) {
            // Fill x_t with the characters at step t in the batch
            //vector<unsigned> id;
            //for (int i = 0; i < bsize; ++i) {
                //id.push_back(t * bsize + i);
                //x_t[i] = isents[id + i][t];
                //x_m[i] = x_mask[id + i][t];
                //if (x_t[i] != *isents[id].rbegin()) chars++; // if x_t is non-EOS, count a char
            //}
            //Expression mask = input(cg, Dim({1}, bsize), x_m);
            // Get embedding
            //Expression i_x_t = pick_batch_elems(emb_x, id);
            // Run a step in the forward encoder
            Expression i_back = fwd_enc_builder.add_input(i_x_t[t]);
            // broadcast is available
            //Expression now_h = cmult(fwd_enc_builder.back(), mask) + cmult(last_h, (1-mask));
            //fwd_enc_builder.set_h(fwd_enc_builder.state(), vector<Expression>(1, now_h));
            //fwd_vectors.push_back(now_h);
            fwd_vectors.push_back(i_back);
            //last_h = now_h;
        }

        // Backward encoder ------------------------------------------------------------------------
        
        // Initialize parameters in bwd_enc_builder
        bwd_enc_builder.new_graph(cg);
        // Initialize the sequence
        bwd_enc_builder.start_new_sequence();

        vector<Expression> bwd_vectors;
        Expression last_h = zeroes(cg, Dim({HIDDEN_DIM}, bsize));
        // Fill x_t with the characters at step t in the batch (in reverse order)
        for (int t = islen - 1; t >= 0; --t) {
            //vector<unsigned> id;
            //for (int i = 0; i < bsize; ++i) {
                //id.push_back(t * bsize + i);
                //x_m[i] = x_mask[id + i][t];
            //}
            //Expression mask = input(cg, Dim({1}, bsize), x_m);
            // Get embedding (could be mutualized with fwd_enc_builder)
            //Expression i_x_t = pick_batch_elems(emb_x, id);
            // Run a step in the forward encoder
            Expression i_back = bwd_enc_builder.add_input(i_x_t[t]);
            // broadcast is available
            Expression now_h = cmult(i_back, mask[t]) + cmult(last_h, (1-mask[t]));
            //rev_enc_builder.set_h(rev_enc_builder.state(), vector<Expression>(1, now_h));
            bwd_vectors.push_back(now_h);
            //bwd_vectors.push_back(i_back);
            last_h = now_h;
        }

        // Collect encodings -----------------------------------------------------------------------
        reverse(bwd_vectors.begin(), bwd_vectors.end()); ///!!!!don't forget
        Expression init = bwd_vectors[0];
        //Expression fwd_encoded = concatenate_cols(fwd_vectors);
        //Expression bwd_encoded = concatenate_cols(bwd_vectors);
        //Expression encoded = concatenate({fwd_encoded, bwd_encoded}); // encoded -> (2*hidden_dim,|F|)

        //vector<float> x_m;
        //for (int i = 0; i < bsize; ++i){
        //    for (int t = 0; t < islen; ++t) {
        //        x_m.push_back(x_mask[id + i][t]);
        //    }
        //}
        //Expression mask = input(cg, Dim({islen, 1}, bsize), x_m);
        vector<Expression> encoded;
        for (int t = 0; t < islen; ++t) {
            encoded.push_back(cmult(concatenate( { fwd_vectors[t], bwd_vectors[t] }), mask[t]));    //bi-lstm encoding
        }

        return { init, concatenate_cols(encoded), concatenate(mask) }; 
    }

    /**
     * Single encode
     */
    vector<Expression> encode(const vector<int>& insent, ComputationGraph & cg) {
        vector<float> mask(insent.size(), 1.);
        auto insents = vector<vector<int>>(1, insent);
        auto masks = vector<vector<float>>(1, mask);
        unsigned chars = 0;
        return encode(insents, masks, 0, 1, chars, cg);
    }
    /**
     * attend
     */
    Expression attend(Expression input_mat, vector<Expression> state, Expression w1dt, Expression xmask, ComputationGraph& cg) {
        // w1dt -> (att,|F|)
        //att_weights=v∗tanh(encodedInput*w1+decoderstate*w2)
        Expression w2 = parameter(cg, attention_w2); // (att,hidden_dim*layers*x) , x=1 for GRU, x=2 for lstm 
        Expression v = parameter(cg, attention_v);   // (1,att)
        Expression w2dt = w2 * concatenate(state); // (att,1)
        Expression unnormalized = transpose(v * tanh(colwise_add(w1dt, w2dt))); // (|F|,1)
        Expression att_weights = softmax(cmult(unnormalized, xmask)); // (|F|,1)
        Expression context = input_mat * att_weights; // (2*hidden_dim,1)
        return context;
    }
    /**
     * Batched decoding
     */
    Expression decode(const vector<Expression> encoded,
                      const vector<vector<int>>& osents,
                      const vector<vector<float>>& y_mask,
                      int id,
                      int bsize,
                      ComputationGraph & cg) {

        Expression hid2hid = parameter(cg, p_hid2hid);
        Expression b_hid = parameter(cg, p_b_hid);

        vector<Expression> init;
        //Expression new_init = pickrange(encoded[0], HIDDEN_DIM, HIDDEN_DIM*2); // GRU
        init.push_back( tanh(affine_transform({b_hid, hid2hid, encoded[0]})) ); 
        //init.push_back( hid2hid*new_init); 

        dec_builder.new_graph(cg);
        dec_builder.start_new_sequence(init);

        Expression readout_allthree = parameter(cg, p_readout_allthree);
        //Expression readout_hidden = parameter(cg, p_readout_hidden);
        Expression readout_offset = parameter(cg, p_readout_offset);

        Expression hid2voc = parameter(cg, p_emb2voc) * parameter(cg, p_hid2emb);
        Expression b_voc = parameter(cg, p_b_voc);

        Expression input_mat = encoded[1]; //concatenate_cols(encoded); // (2*hidden_dim,|F|)
        Expression w1 = parameter(cg, attention_w1);
        Expression w1dt = w1 * input_mat; // (att,|F|)

        Expression last_output_embeddings = zeroes(cg, Dim({INPUT_DIM}, bsize));
        //input(cg, Dim({INPUT_DIM}, bsize), vector<float>(INPUT_DIM * bsize));
        //Expression last_output_embeddings = lookup(cg, p_c, vector<unsigned>(bsize, kSOS));
        //vector<float> y_values(HIDDEN_DIM * 2 * bsize);
        //dec_builder.add_input(concatenate( { input(cg, Dim({ HIDDEN_DIM * 2 }, bsize), y_values), last_output_embeddings }));
    
        const unsigned oslen = osents[id].size();

        vector<unsigned> next_y_t(bsize);
        vector<unsigned> y_t;
        vector<float> y_m;
        vector<Expression> concat_vector(oslen);
        //vector<Expression> i_y_t(oslen);
        //vector<Expression> errs;
        //Expression last_h = input(cg, Dim({HIDDEN_DIM}, bsize), vector<float>(HIDDEN_DIM * bsize));
        
        // Run on output sentence
        for (int t = 0; t < oslen; ++t) {

            Expression context = attend(input_mat, dec_builder.final_s(), w1dt, encoded[2], cg);
            concat_vector[t] = concatenate( { context, last_output_embeddings, dec_builder.back() } );
            // Embed token
            //concat_vector[t] = concatenate( { attend(input_mat, dec_builder.final_s(), w1dt, encoded[2], cg), last_output_embeddings }); // vector -> ( input_dim + hidden_dim * 2)
            // Run decoder step
            //i_y_t[t] = dec_builder.add_input(concat_vector[t]);
            //Expression now_h = cmult(i_y_t, mask) + cmult(last_h, (1-mask));
            //dec_builder.set_h(dec_builder.state(), vector<Expression>(1, now_h));
            //last_h = now_h;
            // Project from output dim to dictionary dimension
            //Expression i_r_t = i_bias + i_R * i_y_t;
            //Expression i_r_t = affine_transform({readout_offset, \
                                                readout_emb_context, concat_vector, \
                                                readout_hidden, i_y_t});
            //Expression maxout = max_dim(reshape(i_r_t, Dim({HIDDEN_DIM/2, 2}, bsize)), 1);
            //Expression prob_vocab = affine_transform({b_voc, hid2voc, i_r_t});
            // Retrieve input
            for (int i = 0; i < bsize; ++i) {
                next_y_t[i] = osents[id + i][t];
                y_t.push_back(osents[id + i][t]);
                y_m.push_back(y_mask[id + i][t]);
                //y_m[i] = y_mask[id + i][t];
            }
            //Expression mask = input(cg, Dim({1}, bsize), y_m);
            // Compute softmax and negative log
            //Expression i_err = pickneglogsoftmax(prob_vocab, next_y_t);
            //errs.push_back(cmult(i_err, mask));
            last_output_embeddings = lookup(cg, p_c, next_y_t);
            dec_builder.add_input( concatenate({ context, last_output_embeddings }) );
        }

        Expression i_r_t = affine_transform({readout_offset,
                                            readout_allthree, concatenate_to_batch(concat_vector)});
                                            //readout_hidden, concatenate_to_batch(i_y_t)});
        Expression prob_vocab = affine_transform({b_voc, hid2voc, i_r_t});
        Expression i_err = pickneglogsoftmax(prob_vocab, y_t);
        Expression mask = input(cg, Dim({1}, bsize * oslen), y_m);

        // Sum loss over batch
        //return sum_batches(sum(errs))/(float)bsize;
        return sum_batches(cmult(i_err, mask))/(float)bsize;
    }
  
    /**
     * find first b smallest element
     */
    vector<pair<int,int>> find_beam_min(const vector<float> &loss, const int &beam_size){
        vector<pair<float,int>> tmp;
        for (unsigned i = 0; i < loss.size(); ++i)
            tmp.push_back( pair<float,int>( loss[i], i ) );
        nth_element(tmp.begin(), tmp.begin() + beam_size, tmp.end());
        vector<pair<int,int>> id;
        for (unsigned i = 0; i < beam_size; i++){
            int tid = tmp[i].second;
            id.push_back( pair<int,int>( tid/TGT_VOCAB_SIZE, tid%TGT_VOCAB_SIZE ) );
        }
        return id;
    }
    /**
     * \brief Generate a sentence from an input sentence
     * \details Samples at each timestep ducring decoding. Possible variations are greedy decoding
     * and beam search for better performance
     * 
     * \param insent Input sentence
     * \param cg Computation Graph
     * 
     * \return Generated sentence (indices in the dictionary)
     */
    vector<unsigned> generate(const vector<int>& insent, int& miss, ComputationGraph & cg) {
        return generate(encode(insent, cg), 3 * insent.size() - 1, miss, cg);
    }
    /**
     * @brief Generate a sentence from an encoding
     * @details You can use this directly to generate random sentences
     * 
     * @param i_nc Input encoding
     * @param oslen Maximum length of output
     * @param cg Computation graph
     * @return Generated sentence (indices in the dictionary)
     */
    vector<unsigned> generate(vector<Expression> encoded, unsigned oslen, int& miss, ComputationGraph & cg, int beam_size = 10) {

        // parameter
        Expression hid2hid = parameter(cg, p_hid2hid);
        Expression b_hid = parameter(cg, p_b_hid);

        Expression readout_allthree = parameter(cg, p_readout_allthree);
        //Expression readout_hidden = parameter(cg, p_readout_hidden);
        Expression readout_offset = parameter(cg, p_readout_offset);

        Expression hid2voc = parameter(cg, p_emb2voc) * parameter(cg, p_hid2emb);
        Expression b_voc = parameter(cg, p_b_voc);

        // encoded info
        Expression input_mat = encoded[1]; // (2*hidden_dim,|F|)
        Expression w1 = parameter(cg, attention_w1);
        Expression w1dt = w1 * input_mat;

        //Expression init1 = pickrange(encoded[0], HIDDEN_DIM, HIDDEN_DIM*2); // GRU
        Expression init = tanh(affine_transform({b_hid, hid2hid, encoded[0]})); 
        //Expression init3 = concatenate_to_batch(vector<Expression>(beam_size, init2)); // copy beam_size times

        // init dec_builder
        dec_builder.new_graph(cg);
        dec_builder.start_new_sequence(vector<Expression>(1, init));

        //Expression last_output_embeddings = input(cg, {INPUT_DIM}, vector<float>(INPUT_DIM)); 
        //Expression last_output_embeddings = lookup(cg, p_c, kSOS);
        //vector<float> x_values(HIDDEN_DIM * 2);
        //dec_builder.add_input(concatenate( { input(cg, { HIDDEN_DIM * 2 }, x_values), last_output_embeddings }));

        // var
        //vector<vector<float>> loss(beam_size);
        vector<vector<float>> sum_loss;
        vector<vector<unsigned>> result;
        vector<vector<unsigned>> fa;
        //for (auto &sent : result) sent.push_back(kSOS);
        //vector<RNNPointer> pGen_builder(beam_size);
        //for (auto &p : pGen_builder) p = dec_builder.state();


        // temp var
        //vector<float> tmp_pre_loss(beam_size);
        //vector<vector<int>> tmp_result(beam_size);
        //vector<RNNPointer> tmp_pGen_builder(beam_size);

        // final var
        vector<vector<unsigned>> final_result;
        vector<float> final_loss;

        Expression last_output_embeddings = zeroes(cg, {INPUT_DIM});

        for (int t = 0; t < oslen; ++t) {
            Expression context = attend(input_mat, dec_builder.final_s(), w1dt, encoded[2], cg);
            //Expression i_y_t = dec_builder.add_input(concat_vector);
            Expression concat_vector = concatenate( {context, last_output_embeddings, dec_builder.back() }); 
            Expression i_r_t = affine_transform({readout_offset, 
                                                readout_allthree, concat_vector});
                                                //readout_hidden, i_y_t});
            //Expression maxout = max_dim(reshape(i_r_t, Dim({HIDDEN_DIM/2, 2})), 1);
            Expression prob_vocab = affine_transform({b_voc, hid2voc, i_r_t});
            Expression i_ydist = log_softmax(prob_vocab);
            vector<float> probs = as_vector(i_ydist.value()); // maybe need joint each batch?
            //assert(probs.size() == beam_size * TGT_VOCAB_SIZE);

            vector<float>& loss = probs;
            for (int j = 0; j < loss.size(); ++j){
                if (t > 0) loss[j] = sum_loss[t-1][j/TGT_VOCAB_SIZE] - probs[j];
                else loss[j] = -probs[j];
            }
            if (t < oslen/6) {
                for (int i = kEOS; i < loss.size(); i += TGT_VOCAB_SIZE) {
                    loss[i] = FLT_MAX; ///trick
                }
            }

            // find_beam_min
            vector<pair<int,int>> id = find_beam_min(loss, beam_size); 
            
            // update info
            sum_loss.push_back(vector<float>());
            result.push_back(vector<unsigned>());
            fa.push_back(vector<unsigned>());
            for (int i = beam_size - 1; i >= 0; i--) { // id.size() == beam_size
                if (id[i].second != kEOS) {
                    sum_loss[t].push_back(loss[id[i].first * TGT_VOCAB_SIZE + id[i].second]);
                    result[t].push_back(id[i].second);
                    fa[t].push_back(id[i].first);
                }
                else {
                    beam_size--;
                    vector<unsigned> tmp = {(unsigned)kEOS};
                    int last = id[i].first;
                    for (int j = t - 1; j >= 0; j--) {
                        tmp.push_back(result[j][last]);
                        last = fa[j][last];
                    }
                    reverse(tmp.begin(), tmp.end());
                    final_result.push_back(tmp);
                    final_loss.push_back(loss[id[i].first * TGT_VOCAB_SIZE + id[i].second]);
                }
            }

            if (beam_size <= 0) break;

            Expression new_context = pick_batch_elems(context, fa[t]);
            dec_builder.set_s(dec_builder.state(), vector<Expression>(1, pick_batch_elems(dec_builder.back(), fa[t])));
            last_output_embeddings = lookup(cg, p_c, result[t]);
            
            dec_builder.add_input( concatenate({new_context, last_output_embeddings}) );

        }

        if (final_result.size() > 0)
            return final_result[min_element(final_loss.begin(),final_loss.end())-final_loss.begin()];
        else { //if (beam_size >= 100) {
            miss++;
            cerr << "cannot find translation in max beam size " << beam_size << endl;
            return result[0];
        }
        //else {
        //    cout << "cannot find translation in beam size " << beam_size << " try " << beam_size*2 << endl;
        //    return generate(encoded, oslen, cg, beam_size*2);
        //}
    }//*/

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & LAYERS & INPUT_DIM & HIDDEN_DIM & ATTENTION_SIZE;
        ar & p_c & p_ec & p_hid2hid & p_b_hid;
        ar & p_readout_allthree & p_readout_offset;
        ar & p_hid2emb & p_emb2voc & p_b_voc;
        ar & attention_w1 & attention_w2 & attention_v;
        ar & dec_builder & bwd_enc_builder & fwd_enc_builder;
    }
};

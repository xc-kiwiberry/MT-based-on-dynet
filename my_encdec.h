#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "dynet/tensor.h"
#include "dynet/io.h"
#include "dynet/param-init.h"
//#include "dynet/gru.h"

#include "my_tools.h"
#include "my_cl-args.h"
#include "my_gru.h"

#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <utility>
#include <unistd.h>
#include <float.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace dynet;

struct EncoderDecoder {
private:
    unsigned INPUT_DIM;
    unsigned HIDDEN_DIM;
    unsigned TGT_VOCAB_SIZE;
    
    GRUBuilder dec_builder;     // GRU
    GRUBuilder fwd_enc_builder;
    GRUBuilder bwd_enc_builder;

    LookupParameter p_c;
    LookupParameter p_ec;  // map input to embedding (used in fwd and rev models)

    Parameter p_hid2hid;
    Parameter p_b_hid;

    Parameter attention_w1;
    Parameter attention_w2;
    Parameter attention_v;

    Parameter p_readout_allthree;
    Parameter p_readout_offset;

    Parameter p_hid2emb;
    Parameter p_emb2voc;
    Parameter p_b_voc;
public:

    EncoderDecoder() {}

    explicit EncoderDecoder(ParameterCollection& model,
                            unsigned LAYERS,
                            unsigned INPUT_DIM,
                            unsigned HIDDEN_DIM,
                            unsigned ATTENTION_SIZE,
                            unsigned SRC_VOCAB_SIZE,
                            unsigned TGT_VOCAB_SIZE,
                            float scale = 0.08) :
        INPUT_DIM(INPUT_DIM), HIDDEN_DIM(HIDDEN_DIM), TGT_VOCAB_SIZE(TGT_VOCAB_SIZE),
        dec_builder(LAYERS, INPUT_DIM + HIDDEN_DIM * 2, HIDDEN_DIM, model),
        fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model),
        bwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model) {

        p_ec = model.add_lookup_parameters(SRC_VOCAB_SIZE, {INPUT_DIM}, ParameterInitUniform(scale));
        p_c = model.add_lookup_parameters(TGT_VOCAB_SIZE, {INPUT_DIM}, ParameterInitUniform(scale));

        p_hid2hid = model.add_parameters( { HIDDEN_DIM, HIDDEN_DIM }, scale);
        p_b_hid = model.add_parameters( { HIDDEN_DIM }, scale);

        attention_w1 = model.add_parameters( { ATTENTION_SIZE, HIDDEN_DIM * 2 }, scale);
        attention_w2 = model.add_parameters( { ATTENTION_SIZE, HIDDEN_DIM * LAYERS }, scale); // GRU
        //attention_w2 = model.add_parameters( { ATTENTION_SIZE, HIDDEN_DIM * LAYERS * 2 }); // LSTM
        attention_v = model.add_parameters( { 1, ATTENTION_SIZE }, scale);

        p_readout_allthree = model.add_parameters( { HIDDEN_DIM, INPUT_DIM + HIDDEN_DIM * 2 + HIDDEN_DIM }, scale);
        p_readout_offset = model.add_parameters( { HIDDEN_DIM }, scale);

        p_hid2emb = model.add_parameters( { INPUT_DIM, HIDDEN_DIM }, scale);
        p_emb2voc = model.add_parameters( { TGT_VOCAB_SIZE, INPUT_DIM }, scale);
        p_b_voc = model.add_parameters( { TGT_VOCAB_SIZE }, scale );
    }

    /**
     * Batched encoding
     */
    vector<Expression> encode(const vector<vector<int>>& isents,
                      const vector<vector<float>>& x_mask,
                      unsigned id,
                      unsigned bsize,
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
            i_x_t[t] = dropout(lookup(cg, p_ec, x_t), 0.2);
            mask[t] = input(cg, Dim({1}, bsize), x_m);
        }

        // Forward encoder -------------------------------------------------------------------------

        // Initialize parameters in fwd_enc_builder
        fwd_enc_builder.new_graph(cg);
        // Initialize the sequence
        fwd_enc_builder.start_new_sequence();

        vector<Expression> fwd_vectors; // fwd_vectors[i] -> (hidden_dim,1)
        
        // Run the forward encoder on the batch
        for (int t = 0; t < islen; ++t) {
            Expression i_back = fwd_enc_builder.add_input(i_x_t[t]);
            fwd_vectors.push_back(i_back);
        }

        // Backward encoder ------------------------------------------------------------------------
        
        // Initialize parameters in bwd_enc_builder
        bwd_enc_builder.new_graph(cg);
        // Initialize the sequence
        bwd_enc_builder.start_new_sequence();

        vector<Expression> bwd_vectors;
        Expression last_h = zeroes(cg, Dim({HIDDEN_DIM}, bsize));
        
        for (int t = islen - 1; t >= 0; --t) {
            Expression i_back = bwd_enc_builder.add_input(i_x_t[t]);
            Expression now_h = cmult(i_back, mask[t]) + cmult(last_h, (1-mask[t]));
            bwd_vectors.push_back(now_h);
            last_h = now_h;
        }

        // Collect encodings -----------------------------------------------------------------------
        reverse(bwd_vectors.begin(), bwd_vectors.end()); ///!!!!don't forget
        Expression init = bwd_vectors[0];
        
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
        return encode(insents, masks, 0, 1, cg);
    }
    /**
     * attend
     */
    Expression attend(const Expression& input_mat, const vector<Expression>& state, const Expression& w1dt, const Expression& xmask, ComputationGraph& cg) {
        // w1dt -> (att,|F|)
        //att_weights=v∗tanh(encodedInput*w1+decoderstate*w2)
        Expression w2 = parameter(cg, attention_w2); // (att,hidden_dim*layers*x) , x=1 for GRU, x=2 for lstm 
        Expression v = parameter(cg, attention_v);   // (1,att)
        Expression w2dt = w2 * concatenate(state); // (att,1)
        Expression unnormalized = transpose(v * tanh(colwise_add(w1dt, w2dt))); // (|F|,1)
        Expression att_weights = softmax(cmult(unnormalized, xmask)); // (|F|,1)
        Expression context = input_mat * att_weights; // (2*hidden_dim,1)
        return reshape(context, Dim({2*HIDDEN_DIM}, context.dim().bd));
    }
    /**
     * Batched decoding
     */
    Expression decode(const vector<Expression>& encoded,
                      const vector<vector<int>>& osents,
                      const vector<vector<float>>& y_mask,
                      int id,
                      int bsize,
                      ComputationGraph & cg) {

        Expression hid2hid = parameter(cg, p_hid2hid);
        Expression b_hid = parameter(cg, p_b_hid);

        vector<Expression> init;
        init.push_back( tanh(affine_transform({b_hid, hid2hid, encoded[0]})) ); 
        
        dec_builder.new_graph(cg);
        dec_builder.start_new_sequence(init);

        Expression readout_allthree = parameter(cg, p_readout_allthree);
        Expression readout_offset = parameter(cg, p_readout_offset);

        Expression hid2voc = parameter(cg, p_emb2voc) * parameter(cg, p_hid2emb);
        Expression b_voc = parameter(cg, p_b_voc);

        Expression input_mat = encoded[1]; //concatenate_cols(encoded); // (2*hidden_dim,|F|)
        Expression w1 = parameter(cg, attention_w1);
        Expression w1dt = w1 * input_mat; // (att,|F|)

        // zero embedding
        Expression last_output_embeddings = zeroes(cg, Dim({INPUT_DIM}, bsize));
        
        const unsigned oslen = osents[id].size();

        vector<unsigned> next_y_t(bsize);
        vector<unsigned> y_t;
        vector<float> y_m;
        vector<Expression> concat_vector(oslen);
        
        // Run on output sentence
        for (int t = 0; t < oslen; ++t) {

            Expression context = attend(input_mat, dec_builder.final_s(), w1dt, encoded[2], cg);
            concat_vector[t] = concatenate( { context, last_output_embeddings, dec_builder.back() } );
            
            //Expression maxout = max_dim(reshape(i_r_t, Dim({HIDDEN_DIM/2, 2}, bsize)), 1);
            // 
            for (int i = 0; i < bsize; ++i) {
                next_y_t[i] = osents[id + i][t];
                y_t.push_back(osents[id + i][t]);
                y_m.push_back(y_mask[id + i][t]);
            }
            // 
            last_output_embeddings = dropout(lookup(cg, p_c, next_y_t), 0.2);
            dec_builder.add_input( concatenate({ context, last_output_embeddings }) );
        }

        Expression i_r_t = affine_transform({readout_offset,
                                            readout_allthree, concatenate_to_batch(concat_vector)});

        Expression prob_vocab = affine_transform({b_voc, hid2voc, dropout(i_r_t, 0.2)});
        Expression i_err = pickneglogsoftmax(prob_vocab, y_t);
        Expression mask = input(cg, Dim({1}, bsize * oslen), y_m);
        
        return cmult(i_err, mask);
    }
  
    /**
     * find first b smallest element
     */
    vector<pair<int,int>> find_beam_min(const vector<float> &loss, const int &beam_size){
        vector< pair<float,int> > tmp;
        for (unsigned i = 0; i < loss.size(); ++i)
            tmp.push_back( pair<float,int>( loss[i], i ) );
        nth_element(tmp.begin(), tmp.begin() + beam_size, tmp.end());
        vector< pair<int,int> > id;
        for (unsigned i = 0; i < beam_size; i++){
            int tid = tmp[i].second;
            id.push_back( pair<int,int>( tid/TGT_VOCAB_SIZE, tid%TGT_VOCAB_SIZE ) );
        }
        return id;
    }

    /**
     * Generate a sentence from an input sentence
     */
    vector<unsigned> generate(const vector<int>& insent, int& miss, ComputationGraph & cg, int beam_size = 10) {
        return generate(encode(insent, cg), 3 * insent.size() - 1, miss, cg, beam_size);
    }

    /**
     * Generate a sentence from an encoding
     */
    vector<unsigned> generate(const vector<Expression>& encoded, 
                            const unsigned& oslen, 
                            int& miss, 
                            ComputationGraph & cg, 
                            int beam_size = 10) {

        // parameter
        Expression hid2hid = parameter(cg, p_hid2hid);
        Expression b_hid = parameter(cg, p_b_hid);

        Expression readout_allthree = parameter(cg, p_readout_allthree);
        Expression readout_offset = parameter(cg, p_readout_offset);

        Expression hid2voc = parameter(cg, p_emb2voc) * parameter(cg, p_hid2emb);
        Expression b_voc = parameter(cg, p_b_voc);

        // encoded info
        Expression input_mat = encoded[1]; // (2*hidden_dim,|F|)
        Expression w1 = parameter(cg, attention_w1);
        Expression w1dt = w1 * input_mat;

        Expression init = tanh(affine_transform({b_hid, hid2hid, encoded[0]})); 
        
        // init dec_builder
        dec_builder.new_graph(cg);
        dec_builder.start_new_sequence(vector<Expression>(1, init));

        // var
        vector<vector<float>> sum_loss;
        vector<vector<unsigned>> result;
        vector<vector<unsigned>> fa;

        // final var
        vector<vector<unsigned>> final_result;
        vector<float> final_loss;

        Expression last_output_embeddings = zeroes(cg, {INPUT_DIM});

        for (int t = 0; t < oslen; ++t) {
            Expression context = attend(input_mat, dec_builder.final_s(), w1dt, encoded[2], cg);
            Expression concat_vector = concatenate( {context, last_output_embeddings, dec_builder.back() }); 
            Expression i_r_t = affine_transform({readout_offset, 
                                                readout_allthree, concat_vector});
            //Expression maxout = max_dim(reshape(i_r_t, Dim({HIDDEN_DIM/2, 2})), 1);
            Expression prob_vocab = affine_transform({b_voc, hid2voc, i_r_t});
            Expression i_ydist = log_softmax(prob_vocab);
            vector<float> probs = as_vector(i_ydist.value()); 
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
        
        // length normalization
        for (int i = 0; i < final_loss.size(); i++){
            final_loss[i] /= final_result[i].size();
        }

        if (final_result.size() > 0) {
            return final_result[min_element(final_loss.begin(),final_loss.end())-final_loss.begin()];
        }
        else { 
            miss++;
            //cerr << "cannot find translation in max beam size " << beam_size << endl;
            return result[0];
        }
    }

    /**
     *  get some samples for MRT training
     */
    vector<vector<int>> sample(const vector<Expression>& encoded, 
                    const unsigned& ref_len,
                    ComputationGraph & cg) {

        // parameter
        Expression hid2hid = parameter(cg, p_hid2hid);
        Expression b_hid = parameter(cg, p_b_hid);

        Expression readout_allthree = parameter(cg, p_readout_allthree);
        Expression readout_offset = parameter(cg, p_readout_offset);

        Expression hid2voc = parameter(cg, p_emb2voc) * parameter(cg, p_hid2emb);
        Expression b_voc = parameter(cg, p_b_voc);

        // encoded info
        Expression input_mat = encoded[1]; // (2*hidden_dim,|F|)
        Expression w1 = parameter(cg, attention_w1);
        Expression w1dt = w1 * input_mat;

        Expression init = tanh(affine_transform({b_hid, hid2hid, encoded[0]})); 
        print_dim(init.dim());
        init = pick_batch_elems(init, vector<unsigned>(params.mrt_sampleSize, 0)); // make {hid} to ({hid},sample_size)
        print_dim(init.dim());

        // init dec_builder
        dec_builder.new_graph(cg);
        dec_builder.start_new_sequence(vector<Expression>(1, init));
        
        // zero embedding
        Expression last_output_embeddings = zeroes(cg, Dim({INPUT_DIM}, params.mrt_sampleSize));
        print_dim(last_output_embeddings.dim());
        vector<vector<int>> hyp_sents(params.mrt_sampleSize, vector<int>());

        unsigned sample_lenth = params.mrt_lenRatio * ref_len;
        for (int t = 0; t < sample_lenth; ++t) {
            Expression context = attend(input_mat, dec_builder.final_s(), w1dt, encoded[2], cg);
            print_dim(context.dim());
            Expression concat_vector = concatenate( {context, last_output_embeddings, dec_builder.back() }); 
            print_dim(concat_vector.dim());
            Expression i_r_t = affine_transform({readout_offset, 
                                                readout_allthree, concat_vector});

            Expression prob_vocab = affine_transform({b_voc, hid2voc, i_r_t});
            vector<float> probs = as_vector((softmax(prob_vocab)).value()); 

            // ramdom sample
            vector<float> randomNum = as_vector(random_uniform(cg, {params.mrt_sampleSize}, 0.0, 1.0).value());
            vector<unsigned> ids;
            for (int i = 0; i < params.mrt_sampleSize; i++){
                for (int j = i*TGT_VOCAB_SIZE; j < (i+1)*TGT_VOCAB_SIZE; j++){
                    randomNum[i] -= probs[j];
                    if (randomNum[i] <= 0){
                        ids.push_back(j % TGT_VOCAB_SIZE);
                        hyp_sents[i].push_back(j % TGT_VOCAB_SIZE);
                        break;
                    }
                }
            }

            last_output_embeddings = lookup(cg, p_c, ids);
            print_dim(last_output_embeddings.dim());
            print_dim(dec_builder.back().dim());
            Expression tmp = concatenate({context, last_output_embeddings});
            print_dim(tmp.dim());
            dec_builder.add_input( tmp );
        }

        return hyp_sents;
    }
};


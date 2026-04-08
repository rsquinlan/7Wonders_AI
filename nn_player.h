#pragma once

#include "game.h"
#include "card.h"
#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

// Wraps an ONNX Runtime session for the trained WondersNet.
// One instance per program run (or per game). Thread-hostile — do not share
// across threads without external locking.
class NNPlayer {
public:
    // Load the ONNX model from disk. Throws std::runtime_error on failure.
    explicit NNPlayer(const std::string& model_path, int num_players);

    // Returns the highest-policy playable card for playerIndex, falling back
    // to the first playable card if the model scores none of them.
    DMAG::Card getBestMoveNN(const std::shared_ptr<DMAG::Game>& game, int playerIndex);

    // Raw inference: fills policy (75 floats) and value (1 float) for a state.
    void infer(const std::vector<float>& state_embedding,
               std::vector<float>& out_policy,
               float&              out_value);

private:
    int num_players_;
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo mem_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

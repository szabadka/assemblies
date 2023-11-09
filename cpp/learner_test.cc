#include "learner.h"

#include <gtest/gtest.h>

constexpr int kLogLevel = 0;

namespace nemo {
namespace {

bool TrainExperiment(uint32_t seed, uint32_t& round) {
  LearnerParams params;
  LearnerBrain learner(params, seed);
  learner.SetLogLevel(kLogLevel);
  round = 0;
  bool success = false;
  for (; !success && round < 20; ++round) {
    //if (kLogLevel > 0) {
      printf("Round %d\n", round);
      //}
    for (uint32_t i = 0; i < params.num_nouns; ++i) {
      for (uint32_t j = 0; j < params.num_verbs; ++j) {
        learner.ParseIndexedSentence(i, params.num_nouns + j);
      }
    }
    success = learner.TestAllProduction();
  }
  learner.LogGraphStats();
  return success;
}

TEST(LearnerTest, Simple) {
  uint32_t total_rounds = 0;
  const int num_reruns = 3;
  for (int rerun = 0; rerun < num_reruns; ++rerun) {
    uint32_t round;
    bool success = TrainExperiment(7774 + 0 + rerun * 17, round);
    EXPECT_TRUE(success);
    if (success) {
      printf("Success after %d rounds\n", round);
    }
    total_rounds += round;
  }
  printf("Average rounds: %f\n", total_rounds * 1.0f / num_reruns);
}

}  // namespace
}  // namespace nemo

#include "learner.h"

#include <random>

#include <gtest/gtest.h>

constexpr int kLogLevel = 0;

namespace nemo {
namespace {

bool TrainExperiment(uint32_t seed, size_t& num) {
  LearnerParams params;
  LearnerBrain learner(params, seed);
  learner.SetLogLevel(kLogLevel);
  num = 0;
  const bool use_context = false; //params.context_areas > 0;
  bool success = false;
  const size_t n = params.num_nouns;
  const size_t v = params.num_verbs;
  size_t max_num_sentences = 100 * (n + v);
  std::mt19937 rng(1777);
  std::uniform_int_distribution<> un(0, n - 1);
  std::uniform_int_distribution<> uv(0, v - 1);
  for (; num < max_num_sentences; ++num) {
    if (num > 0 && num % 10 == 0) {
      const bool verbose = num % 100 == 0;
      if (verbose) {
        printf("\nNumber of sentences trained: %zu\n", num);
      }
      if (learner.TestConvergence(use_context, verbose)) {
        success = true;
        break;
      }
      if (verbose) {
        learner.LogGraphStats();
      }
    }
    learner.ParseIndexedSentence(un(rng), n + uv(rng));
  }
  learner.LogGraphStats();
#if 0
  printf("\n\n");
  learner.SetLogLevel(0);
  learner.TestWordProduction(0, false, 2);
  printf("\n");
  learner.TestWordProduction(n - 1, false, 2);
  printf("\n");
  learner.TestWordProduction(n, false, 2);
  printf("\n");
  learner.TestWordProduction(n + v - 1, false, 2);
  printf("\n\n");
  learner.AnalyseAssemblies(kVisualArea, kNounArea, 0, n);
  learner.AnalyseAssemblies(kPhonArea, kNounArea, 0, n);
  learner.AnalyseAssemblies(kPhonArea, kNounArea, n, n + v);
  learner.AnalyseAssemblies(kMotorArea, kVerbArea, 0, n);
  learner.AnalyseAssemblies(kPhonArea, kVerbArea, n, n + v);
  learner.AnalyseAssemblies(kPhonArea, kVerbArea, 0, n);
  learner.SetLogLevel(kLogLevel);
#endif
  printf("%s after %zu sentences\n", success ? "Success" : "Failure", num);
  learner.TestConvergence(use_context, true);
  return success;
}

TEST(LearnerTest, Simple) {
  uint32_t total = 0;
  const int num_reruns = 1;
  for (int rerun = 0; rerun < num_reruns; ++rerun) {
    size_t num;
    bool success = TrainExperiment(7774 + 0 + rerun * 17, num);
    EXPECT_TRUE(success);
    total += num;
  }
  printf("Average sentences: %f\n", total * 1.0f / num_reruns);
}

}  // namespace
}  // namespace nemo

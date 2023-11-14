#include "learner.h"

#include <gtest/gtest.h>

namespace nemo {
namespace {

class LearnerTestParam : public testing::TestWithParam<uint32_t> {};

TEST_P(LearnerTestParam, Small) {
  LearnerParams params;
  params.beta = 0.06;
  params.max_weight = 1000.0;
  params.lex_n = 100000;
  params.num_nouns = 2;
  params.num_verbs = 2;
  LearnerBrain learner(params, GetParam());
  for (size_t i = 0; i < 60; ++i) {
    learner.ParseRandomSentence();
  }
  learner.LogGraphStats();
  EXPECT_TRUE(learner.TestAllWords(
      /*use_context=*/false, /*test_cross_input=*/false, /*log_level=*/1));
}

#if 0
TEST_P(LearnerTestParam, SmallWithContext) {
  LearnerParams params;
  params.beta = 0.1;
  params.max_weight = 10000.0;
  params.lex_n = 100000;
  params.num_nouns = 2;
  params.num_verbs = 2;
  params.context_areas = 5;
  LearnerBrain learner(params, GetParam());
  for (size_t i = 0; i < 60; ++i) {
    learner.ParseRandomSentence();
  }
  learner.LogGraphStats();
  EXPECT_TRUE(learner.TestAllWords(
      /*use_context=*/true, /*test_cross_input=*/false, /*log_level=*/1));
}
#endif

TEST_P(LearnerTestParam, Medium) {
  LearnerParams params;
  params.beta = 0.1;
  params.max_weight = 10000.0;
  params.lex_n = 1000000;
  params.lex_k = 50;
  params.num_nouns = 20;
  params.num_verbs = 20;
  params.projection_rounds = 2;
  LearnerBrain learner(params, GetParam());
  for (size_t i = 0; i < 1000; ++i) {
    learner.ParseRandomSentence();
  }
  learner.LogGraphStats();
  EXPECT_TRUE(learner.TestAllWords(
      /*use_context=*/false, /*test_cross_input=*/false, /*log_level=*/1));
}

#if 0
TEST_P(LearnerTestParam, Large) {
  LearnerParams params;
  params.beta = 0.1;
  params.max_weight = 10000.0;
  params.lex_n = 1000000;
  params.lex_k = 50;
  params.num_nouns = 400;
  params.num_verbs = 400;
  params.projection_rounds = 2;
  LearnerBrain learner(params, GetParam());
  for (size_t i = 0; i < 10000; ++i) {
    learner.ParseRandomSentence();
  }
  learner.LogGraphStats();
  EXPECT_TRUE(learner.TestAllWords(
      /*use_context=*/false, /*test_cross_input=*/false, /*log_level=*/1));
}
#endif

INSTANTIATE_TEST_SUITE_P(LearnerTest, LearnerTestParam,
                         testing::Values(7, 77, 777, 7777, 77777));

}  // namespace
}  // namespace nemo

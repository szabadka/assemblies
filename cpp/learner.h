#ifndef NEMO_LEARNER_H_
#define NEMO_LEARNER_H_

#include <stdint.h>

#include <set>
#include <string>
#include <vector>

#include "brain.h"

namespace nemo {

struct LearnerParams {
  float p = 0.05;
  float beta = 0.02;
  uint32_t phon_k = 100;
  uint32_t visual_k = 100;
  uint32_t motor_k = 100;
  uint32_t lex_n = 20000;
  uint32_t lex_k = 100;
  uint32_t num_nouns = 20;
  uint32_t num_verbs = 20;
  uint32_t context_areas = 0;
  uint32_t context_k = 10;
  uint32_t context_delay = 0;
  uint32_t projection_rounds = 2;
};

constexpr const char* kPhonArea = "PHON";
constexpr const char* kVisualArea = "VISUAL";
constexpr const char* kMotorArea = "MOTOR";
constexpr const char* kNounArea = "NOUN";
constexpr const char* kVerbArea = "VERB";

class LearnerBrain : public Brain {
 public:
  LearnerBrain(const LearnerParams& params, uint32_t seed);

  bool ParseIndexedSentence(uint32_t noun_index, uint32_t verb_index);

  // property P test
  bool TestWordProduction(uint32_t word, bool use_context = false);
  bool TestAllProduction(bool use_context = false);

  // property Q test
  bool TestWordAssembly(uint32_t word);

 private:
  static std::string ContextAreaName(uint32_t i) {
    return std::string("CONTEXT") + std::to_string(i);
  }

  bool ActivateWord(uint32_t word, bool use_context);
  bool ActivateSentence(uint32_t noun_index, uint32_t verb_index,
                        bool use_context);
  void ProjectStar(bool mutual_inhibition = false);
  void ClearContexts();
  float TotalInput(const std::string& from, const std::string& to) const;

  LearnerParams params_;
  uint32_t lex_size_;
  std::vector<uint32_t> context_map_;
  uint32_t sentences_parsed_;
  std::set<uint32_t> noun_perma_active_;
  std::set<uint32_t> verb_perma_active_;
};

}  // namespace nemo

#endif  // NEMO_LEARNER_H_

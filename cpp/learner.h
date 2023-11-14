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
  float beta = 0.06;
  float max_weight = 1000.0;
  uint32_t phon_k = 100;
  uint32_t visual_k = 100;
  uint32_t motor_k = 100;
  uint32_t lex_n = 100000;
  uint32_t lex_k = 100;
  uint32_t num_nouns = 10;
  uint32_t num_verbs = 10;
  uint32_t context_areas = 0;
  uint32_t context_k = 20;
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

  bool ParseRandomSentence();

  bool TestWord(uint32_t word, bool use_context, bool test_cross_input,
                int log_level = 0);
  bool TestAllWords(bool use_context, bool test_cross_input, int log_level = 0);

  void AnalyseInput(const std::string& from,
                    const std::string& to,
                    float& total_w,
                    size_t& num_synapses,
                    size_t& num_sat_weights) const;
  void AnalyseAssemblies(const std::string& from,
                         const std::string& to,
                         size_t start, size_t end);

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

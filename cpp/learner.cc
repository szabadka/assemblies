#include "learner.h"

#include <map>
#include <random>
#include <set>

#include "brain_util.h"

namespace nemo {

LearnerBrain::LearnerBrain(const LearnerParams& params, uint32_t seed) :
    Brain(params.p, params.beta, seed), params_(params) {
  lex_size_ = params_.num_nouns + params_.num_verbs;

  AddArea(kPhonArea, lex_size_ * params_.phon_k, params_.phon_k,
          /*recurrent=*/false, /*is_explicit=*/true);
  AddArea(kVisualArea, params_.num_nouns * params_.visual_k, params_.visual_k,
          /*recurrent=*/false, /*is_explicit=*/true);
  AddArea(kMotorArea, params_.num_verbs * params_.motor_k, params_.motor_k,
          /*recurrent=*/false, /*is_explicit=*/true);
  for (uint32_t i = 0; i < params_.context_areas; ++i) {
    AddArea(ContextAreaName(i), lex_size_ * params_.context_k,
            params_.context_k, /*recurrent=*/false, /*is_explicit=*/true);
  }
  AddArea(kNounArea, params_.lex_n, params_.lex_k,
          /*recurrent=*/true, /*is_explicit=*/true);
  AddArea(kVerbArea, params_.lex_n, params_.lex_k,
          /*recurrent=*/true, /*is_explicit=*/true);

  AddFiber(kPhonArea, kNounArea, /*bidirectional=*/true);
  AddFiber(kPhonArea, kVerbArea, /*bidirectional=*/true);
  AddFiber(kVisualArea, kNounArea, /*bidirectional=*/true);
  AddFiber(kMotorArea, kVerbArea, /*bidirectional=*/true);
  for (uint32_t i = 0; i < params_.context_areas; ++i) {
    AddFiber(ContextAreaName(i), kNounArea, /*bidirectional=*/false);
    AddFiber(ContextAreaName(i), kVerbArea, /*bidirectional=*/false);
  }

  if (params_.context_areas > 0) {
    context_map_.resize(lex_size_);
    std::uniform_int_distribution<> u(0, params_.context_areas - 1);
    for (uint32_t i = 0; i < lex_size_; ++i) {
      context_map_[i] = u(rng_);
    }
  }
  sentences_parsed_ = 0;
}

bool LearnerBrain::ParseIndexedSentence(uint32_t noun_index,
                                        uint32_t verb_index) {
  if (log_level_ > 0) {
    printf("Parsing sentence %u %u\n", noun_index, verb_index);
  }
  const bool use_context = sentences_parsed_ > params_.context_delay;
  if (!ActivateSentence(noun_index, verb_index, use_context)) {
    return false;
  }
  ActivateArea(kPhonArea, noun_index);
  ProjectStar();
  ActivateArea(kPhonArea, verb_index);
  ProjectStar();
  ClearContexts();
  ++sentences_parsed_;
  return true;
}

uint32_t LearnerBrain::TestWordProduction(uint32_t word, bool use_context,
                                          int log_level) {
  if (log_level_ > 0) {
    printf("Testing word %u production\n", word);
  }
  GetArea(kPhonArea).is_fixed = false;
  if (!ActivateWord(word, use_context)) {
    return false;
  }
  const bool is_noun = word < params_.num_nouns;
  const char* context_area = is_noun ? kVisualArea : kMotorArea;
  const char* lex_area = is_noun ? kNounArea : kVerbArea;
  Project({{context_area, {lex_area}}}, 1, false);
  if (log_level > 1) {
    LogActivated(lex_area);
    LogInput(context_area, lex_area);
  }
  Project({{lex_area, {kPhonArea}}}, 1, false);
  const Area& phon = GetArea(kPhonArea);
  uint32_t offset = word * phon.k;
  uint32_t overlap = 0;
  for (uint32_t neuron : phon.activated) {
    if (neuron >= offset && neuron < offset + phon.k) ++overlap;
  }
  if (log_level > 1) {
    LogActivated(kPhonArea);
    LogInput(lex_area, kPhonArea);
  }
  if (log_level > 0) {
    printf("Overlap: %u (%.0f%%)\n", overlap, overlap * 100.0f / phon.k);
  }
  return overlap;
}

bool LearnerBrain::TestAllProduction(bool use_context, bool verbose) {
  size_t num_success = 0;
  size_t sum_overlap = 0;
  for (uint32_t w = 0; w < lex_size_; ++w) {
    uint32_t overlap = TestWordProduction(w, use_context);
    sum_overlap += overlap;
    if (overlap >= 0.75 * params_.phon_k) {
      ++num_success;
    } else if (!verbose) {
      return false;
    }
  }
  if (verbose) {
    printf("Number of words satisfying property P: %zu (%.0f%%), "
           "average overlap: %.1f%%\n",
           num_success, num_success * 100.0f / lex_size_,
           sum_overlap * 100.0f / params_.phon_k / lex_size_);
  }
  return num_success == lex_size_;
}

bool LearnerBrain::TestWordAssembly(uint32_t word) {
  const bool is_noun = word < params_.num_nouns;
  ActivateArea(kPhonArea, word);
  Project({{kPhonArea, {kNounArea, kVerbArea}}}, 1, false);
  const char* area0 = is_noun ? kNounArea : kVerbArea;
  const char* area1 = is_noun ? kVerbArea : kNounArea;
  const float input0 = TotalInput(kPhonArea, area0);
  const float input1 = TotalInput(kPhonArea, area1);
  if (input0 < 2.0f * input1) {
    return false;
  }
#if 1
  const auto& prev_assembly0 = GetArea(area0).activated;
  const auto& prev_assembly1 = GetArea(area1).activated;
  Project({{kNounArea, {kNounArea}}, {kVerbArea, {kVerbArea}}}, 1, false);
  const size_t common0 = NumCommon(prev_assembly0, GetArea(area0).activated);
  const size_t common1 = NumCommon(prev_assembly1, GetArea(area1).activated);
  printf("common0 = %zu  common1 = %zu\n", common0, common1);
  if (common0 < 0.75f * params_.lex_k) {
    ///printf("common0 = %zu\n", common0);
    return false;
  }
  if (common1 > 0.6f * params_.lex_k) {
    //printf("common1 = %zu\n", common1);
    return false;
  }
#endif
  return true;
}

bool LearnerBrain::TestAllWordAssemblies(bool verbose) {
  size_t num_success = 0;
  for (uint32_t w = 0; w < lex_size_; ++w) {
    bool success = TestWordAssembly(w);
    if (success) {
      ++num_success;
    } else if (!verbose) {
      return false;
    }
  }
  if (verbose) {
    printf("Number of words satisfying property Q: %zu (%.0f%%)\n",
           num_success, num_success * 100.0f / lex_size_);
  }
  return num_success == lex_size_;
}

bool LearnerBrain::TestConvergence(bool use_context, bool verbose) {
  if (!TestAllProduction(use_context, verbose)) {
    return false;
  }
  if (!TestAllWordAssemblies(verbose)) {
    return false;
  }
  return true;
}

bool LearnerBrain::ActivateWord(uint32_t word, bool use_context) {
  if (word >= lex_size_) {
    fprintf(stderr, "Invalid word index %u\n", word);
    return false;
  }
  if (word < params_.num_nouns) {
    ActivateArea(kVisualArea, word);
  } else {
    ActivateArea(kMotorArea, word - params_.num_nouns);
  }
  if (use_context && !context_map_.empty()) {
    ActivateArea(ContextAreaName(context_map_[word]), word);
  }
  return true;
}

bool LearnerBrain::ActivateSentence(uint32_t noun_index, uint32_t verb_index,
                                    bool use_context) {
  if (noun_index >= params_.num_nouns) {
    fprintf(stderr, "Invalid noun index %u\n", noun_index);
    return false;
  }
  if (verb_index < params_.num_nouns || verb_index >= lex_size_) {
    fprintf(stderr, "Invalid verb index %u\n", verb_index);
    return false;
  }
  uint32_t motor_index = verb_index - params_.num_nouns;
  ActivateArea(kVisualArea, noun_index);
  ActivateArea(kMotorArea, motor_index);
  if (use_context && !context_map_.empty()) {
    const std::string noun_context = ContextAreaName(context_map_[noun_index]);
    const std::string verb_context = ContextAreaName(context_map_[verb_index]);
    if (noun_context == verb_context) {
      std::uniform_int_distribution<> u(0, 1);
      if (u(rng_)) {
        ActivateArea(noun_context, noun_index);
      } else {
        ActivateArea(verb_context, verb_index);
      }
    } else {
      ActivateArea(noun_context, noun_index);
      ActivateArea(verb_context, verb_index);
    }
  }
  return true;
}

void UpdatePermaActive(const std::vector<uint32_t>& active,
                       std::set<uint32_t>& perma) {
  if (perma.empty()) {
    for (auto n : active) perma.insert(n);
  } else if (!active.empty()) {
    std::set<uint32_t> lookup(active.begin(), active.end());
    std::vector<uint32_t> tmp(perma.begin(), perma.end());
    for (auto n : tmp) {
      if (lookup.find(n) == lookup.end()) perma.erase(n);
    }
    if (perma.empty()) perma.insert(0xffff);
  }
}

void LearnerBrain::ProjectStar(bool mutual_inhibition) {
  if (log_level_ > 0) {
    printf("ProjectStar\n");
  }
  ProjectMap project_map = {{kPhonArea, {kNounArea, kVerbArea}}};
  const Area& visual = GetArea(kVisualArea);
  const Area& motor = GetArea(kMotorArea);
  if (!visual.activated.empty()) {
    project_map[kVisualArea].push_back(kNounArea);
  }
  if (!motor.activated.empty()) {
    project_map[kMotorArea].push_back(kVerbArea);
  }
  for (uint32_t i = 0; i < params_.context_areas; ++i) {
    const std::string& context_name = ContextAreaName(i);
    if (!GetArea(context_name).activated.empty()) {
      project_map[context_name] = {kNounArea, kVerbArea};
    }
  }
  Project(project_map, 1);
#if 0
  if (sentences_parsed_ > 8) {
    UpdatePermaActive(GetArea(kNounArea).activated, noun_perma_active_);
  }
#endif
  project_map[kNounArea] = {kNounArea, kPhonArea};
  project_map[kVerbArea] = {kVerbArea, kPhonArea};
  if (!visual.activated.empty()) {
    project_map[kNounArea].push_back(kVisualArea);
  }
  if (!motor.activated.empty()) {
    project_map[kVerbArea].push_back(kMotorArea);
  }
  if (mutual_inhibition) {
    float noun_input = (TotalInput(kPhonArea, kNounArea) +
                        TotalInput(kVisualArea, kNounArea));
    float verb_input = (TotalInput(kPhonArea, kVerbArea) +
                        TotalInput(kMotorArea, kVerbArea));
    if (noun_input > verb_input) {
      project_map.erase(kVerbArea);
      project_map[kPhonArea] = {kNounArea};
    } else {
      project_map.erase(kNounArea);
      project_map[kPhonArea] = {kVerbArea};
    }
  }
  Project(project_map, params_.projection_rounds);
#if 0
  if (sentences_parsed_ > 8) {
    UpdatePermaActive(GetArea(kNounArea).activated, noun_perma_active_);
    printf("Noun perma actives: ");
    for (auto n : noun_perma_active_) printf(" %u", n);
    printf("\n");
  }
#endif
}

void LearnerBrain::ClearContexts() {
  GetArea(kMotorArea).activated.clear();
  GetArea(kVisualArea).activated.clear();
  for (uint32_t i = 0; i < params_.context_areas; ++i) {
    GetArea(ContextAreaName(i)).activated.clear();
  }
}

float LearnerBrain::TotalInput(const std::string& from,
                               const std::string& to) const {
  const Area& area_from = GetArea(from);
  const Area& area_to = GetArea(to);
  const Fiber& fiber = GetFiber(from, to);
  std::set<uint32_t> to_activated(area_to.activated.begin(),
                                  area_to.activated.end());
  float total_w = 0.0f;
  for (uint32_t neuron : area_from.activated) {
    const auto& connections = fiber.outgoing_connections[neuron];
    const auto& weights = fiber.outgoing_weights[neuron];
    for (size_t i = 0; i < connections.size(); ++i) {
      if (to_activated.find(connections[i]) != to_activated.end()) {
        total_w += weights[i];
      }
    }
  }
  return total_w;
}

void LearnerBrain::AnalyseInput(const std::string& from,
                                const std::string& to,
                                float& total_w,
                                size_t& num_synapses,
                                size_t& num_sat_weights) const {
  const Area& area_from = GetArea(from);
  const Area& area_to = GetArea(to);
  const Fiber& fiber = GetFiber(from, to);
  std::set<uint32_t> to_activated(area_to.activated.begin(),
                                  area_to.activated.end());
  for (uint32_t neuron : area_from.activated) {
    const auto& connections = fiber.outgoing_connections[neuron];
    const auto& weights = fiber.outgoing_weights[neuron];
    for (size_t i = 0; i < connections.size(); ++i) {
      if (to_activated.find(connections[i]) != to_activated.end()) {
        total_w += weights[i];
        ++num_synapses;
        if (weights[i] == max_weight_) {
          ++num_sat_weights;
        }
      }
    }
  }
}

void LearnerBrain::LogInput(const std::string& from,
                            const std::string& to) const {
  float total_w = 0;
  size_t num_synapses = 0;
  size_t num_sat_weights = 0;
  AnalyseInput(from, to, total_w, num_synapses, num_sat_weights);
  printf("%s -> %s total input: %f num synapses: %zu num saturated: %zu\n",
         from.c_str(), to.c_str(), total_w, num_synapses, num_sat_weights);
}

void LearnerBrain::AnalyseAssemblies(const std::string& from,
                                     const std::string& to,
                                     size_t start, size_t end) {
  const Area& area_from = GetArea(from);
  const Area& area_to = GetArea(to);
  std::vector<std::set<uint32_t>> all_assemblies;
  std::map<uint32_t, uint32_t> mult;
  float total_w = 0;
  size_t num_synapses = 0;
  size_t num_sat_weights = 0;
  for (uint32_t index = start; index < end; ++index) {
    ActivateArea(from, index);
    Project({{from, {to}}}, 1, false);
    LogInput(from, to);
    AnalyseInput(from, to, total_w, num_synapses, num_sat_weights);
    const auto& activated = area_to.activated;
    std::set<uint32_t> assembly(activated.begin(), activated.end());
    all_assemblies.emplace_back(std::move(assembly));
    for (auto n : activated) {
      ++mult[n];
    }
  }
  printf("Total saturated weights for all assemblies: %zu\n", num_sat_weights);
  std::map<uint32_t, uint32_t> mult_histo;
  for (const auto& [key, value] : mult) {
    ++mult_histo[value];
  }
  size_t total = 0;
  for (const auto& [key, value] : mult_histo) {
    total += key * value;
    printf("Neurons appearing in %2u assemblies: %u\n", key, value);
  }
  printf("total = %zu\n", total);
}

}  // namespace nemo

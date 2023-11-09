#include "learner.h"

#include <random>
#include <set>

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
          /*recurrent=*/true, /*is_explicit=*/false);
  AddArea(kVerbArea, params_.lex_n, params_.lex_k,
          /*recurrent=*/true, /*is_explicit=*/false);

  AddFiber(kPhonArea, kNounArea, /*bidirectional=*/true);
  AddFiber(kPhonArea, kVerbArea, /*bidirectional=*/true);
  AddFiber(kVisualArea, kNounArea, /*bidirectional=*/true);
  AddFiber(kMotorArea, kVerbArea, /*bidirectional=*/true);
  for (uint32_t i = 0; i < params_.context_areas; ++i) {
    AddFiber(kNounArea, ContextAreaName(i), /*bidirectional=*/true);
    AddFiber(kVerbArea, ContextAreaName(i), /*bidirectional=*/true);
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

bool LearnerBrain::TestWordProduction(uint32_t word, bool use_context) {
  if (log_level_ > 0) {
    printf("Testing word %u production\n", word);
  }
  GetArea(kPhonArea).is_fixed = false;
  if (!ActivateWord(word, use_context)) {
    return false;
  }
  if (word < params_.num_nouns) {
    Project({{kVisualArea, {kNounArea}}}, 1, false);
    Project({{kNounArea, {kPhonArea}}}, 1, false);
  } else {
    Project({{kMotorArea, {kVerbArea}}}, 1, false);
    Project({{kVerbArea, {kPhonArea}}}, 1, false);
  }
  const Area& phon = GetArea(kPhonArea);
  uint32_t offset = word * phon.k;
  uint32_t overlap = 0;
  for (uint32_t neuron : phon.activated) {
    if (neuron >= offset && neuron < offset + phon.k) ++overlap;
  }
  return overlap >= 0.75 * phon.k;
}

bool LearnerBrain::TestAllProduction(bool use_context) {
  for (uint32_t w = 0; w < lex_size_; ++w) {
    if (!TestWordProduction(w, use_context)) {
      return false;
    }
  }
  return true;
}

bool LearnerBrain::TestWordAssembly(uint32_t word) {
  return false;
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
    for (auto s : fiber.outgoing_synapses[neuron]) {
      if (to_activated.find(s.neuron) != to_activated.end()) {
        total_w += s.weight;
      }
    }
  }
  return total_w;
}

}  // namespace nemo

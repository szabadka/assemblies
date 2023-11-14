#include "learner.h"

#include <map>
#include <random>
#include <set>

#include "brain_util.h"

namespace nemo {

LearnerBrain::LearnerBrain(const LearnerParams& params, uint32_t seed) :
    Brain(params.p, params.beta, params.max_weight, seed), params_(params) {
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

bool LearnerBrain::ParseRandomSentence() {
  std::uniform_int_distribution<> un(0, params_.num_nouns - 1);
  std::uniform_int_distribution<> uv(0, params_.num_verbs - 1);
  const uint32_t noun_index = un(rng_);
  const uint32_t verb_index = uv(rng_) + params_.num_nouns;
  return ParseIndexedSentence(noun_index, verb_index);
}

bool TestAssemblyReproduction(
    Brain& brain, const LearnerParams& params,
    uint32_t word, const char* start_area,
    const std::vector<uint32_t>& lex_assembly,
    bool use_context, int log_level) {
  const bool is_noun = word < params.num_nouns;
  const char* lex_area = is_noun ? kNounArea : kVerbArea;
  const char* ctx_area = is_noun ? kVisualArea : kMotorArea;
  const uint32_t ctx_k = brain.GetArea(ctx_area).k;
  const uint32_t ctx_word = word - (is_noun ? 0 : params.num_nouns);

  brain.GetArea(kPhonArea).is_fixed = false;
  brain.GetArea(ctx_area).is_fixed = false;
  brain.Project({{lex_area, {lex_area, kPhonArea, ctx_area}}}, 1, false);
  const size_t lex_overlap =
      NumCommon(lex_assembly, brain.GetArea(lex_area).activated);
  if (log_level > 0) {
    printf("Word %d %s->%s->%s assembly stability: %zu (%.0f%%)\n",
           word, start_area, lex_area, lex_area,
           lex_overlap, lex_overlap * 100.0f / params.lex_k);
  }
  if (lex_overlap < 0.75f * params.lex_k) {
    return false;
  }
  size_t ctx_index, ctx_overlap;
  brain.ReadAssembly(ctx_area, ctx_index, ctx_overlap);
  if (log_level > 0) {
    printf("Word %d %s->%s->%s index: %zu overlap: %zu (%.0f%%)\n",
           word, start_area, lex_area, ctx_area, ctx_index, ctx_overlap,
           ctx_overlap * 100.0f / ctx_k);
  }
  if (ctx_index != ctx_word || ctx_overlap < 0.75f * ctx_k) {
    return false;
  }
  size_t phon_index, phon_overlap;
  brain.ReadAssembly(kPhonArea, phon_index, phon_overlap);
  if (log_level > 0) {
    printf("Word %d %s->%s->%s index: %zu overlap: %zu (%.0f%%)\n",
           word, start_area, lex_area, kPhonArea, phon_index, phon_overlap,
           phon_overlap * 100.0f / params.phon_k);
  }
  if (phon_index != word || phon_overlap < 0.75f * params.phon_k) {
    return false;
  }
  return true;
}

bool LearnerBrain::TestWord(uint32_t word, bool use_context,
                            bool test_cross_input, int log_level) {
  int prev_log_level = log_level_;
  if (log_level > 1) {
    printf("Testing word %u production\n", word);
    SetLogLevel(std::max(log_level, prev_log_level));
  }
  const bool is_noun = word < params_.num_nouns;
  const char* lex_area = is_noun ? kNounArea : kVerbArea;
  const char* ctx_area = is_noun ? kVisualArea : kMotorArea;
  const uint32_t ctx_word = word - (is_noun ? 0 : params_.num_nouns);

  ActivateArea(ctx_area, ctx_word);
  ProjectMap proj_map = {{ctx_area, {lex_area}}};
  if (use_context && !context_map_.empty()) {
    const std::string& name = ContextAreaName(context_map_[word]);
    ActivateArea(name, word);
    proj_map[name] = { lex_area };
  }
  Project(proj_map, 1, false);
  const auto ctx_lex_assembly = GetArea(lex_area).activated;
  if (!TestAssemblyReproduction(*this, params_, word, ctx_area,
                                ctx_lex_assembly, use_context, log_level)) {
    return false;
  }

  ActivateArea(kPhonArea, word);
  Project({{kPhonArea, {lex_area}}}, 1, false);
  const auto phon_lex_assembly = GetArea(lex_area).activated;
  if (!TestAssemblyReproduction(*this, params_, word, kPhonArea,
                                phon_lex_assembly,
                                use_context, log_level)) {
    return false;
  }

  const size_t lex_assembly_overlap =
      NumCommon(ctx_lex_assembly, phon_lex_assembly);
  if (log_level > 0) {
    printf("Word %d %s->%s and %s->%s assembly overlap: %zu (%.0f%%)\n",
           word, ctx_area, lex_area, kPhonArea, lex_area,
           lex_assembly_overlap, lex_assembly_overlap * 100.0f / params_.lex_k);
  }
#if 0
  // TODO(szabadka) Debug why this fails sometimes.
  if (lex_assembly_overlap < 0.75f * params_.lex_k) {
    return false;
  }
#endif

  const char* lex1_area = is_noun ? kVerbArea : kNounArea;
  const char* ctx1_area = is_noun ? kMotorArea : kVisualArea;
  const uint32_t ctx1_k = GetArea(ctx1_area).k;

  ActivateArea(kPhonArea, word);
  Project({{kPhonArea, {kNounArea, kVerbArea}}}, 1, false);

  const float input = TotalInput(kPhonArea, lex_area);
  const float cross_input = TotalInput(kPhonArea, lex1_area);
  if (log_level > 0) {
    printf("Word %d %s->%s total input: %f\nWord %d %s->%s total input: %f\n",
           word, kPhonArea, lex_area, input,
           word, kPhonArea, lex1_area, cross_input);
  }
  if (input < 2.0f * cross_input) {
    return false;
  }

  if (test_cross_input) {
    ActivateArea(kPhonArea, word);
    Project({{kPhonArea, {lex1_area}}}, 1, false);
    const auto lex1_assembly = GetArea(lex1_area).activated;
    GetArea(kPhonArea).is_fixed = false;
    GetArea(ctx1_area).is_fixed = false;
    Project({{lex1_area, {lex1_area, kPhonArea, ctx1_area}}}, 1, false);
    const size_t lex1_overlap =
        NumCommon(lex1_assembly, GetArea(lex1_area).activated);
    if (log_level > 0) {
      printf("Word %d %s->%s->%s assembly stability: %zu (%.0f%%)\n",
             word, kPhonArea, lex1_area, lex1_area,
             lex1_overlap, lex1_overlap * 100.0f / params_.lex_k);
    }
    if (lex1_overlap > 0.7f * params_.lex_k) {
      return false;
    }
    size_t ctx1_index, ctx1_overlap;
    ReadAssembly(ctx1_area, ctx1_index, ctx1_overlap);
    if (log_level > 0) {
      printf("Word %d %s->%s->%s index: %zu overlap: %zu (%.0f%%)\n",
             word, kPhonArea, lex1_area, ctx1_area, ctx1_index, ctx1_overlap,
             ctx1_overlap * 100.0f / ctx1_k);
    }
    if (ctx1_overlap > 0.7f * ctx1_k) {
      return false;
    }
    size_t phon_index, phon_overlap;
    ReadAssembly(kPhonArea, phon_index, phon_overlap);
    if (log_level > 0) {
      printf("Word %d %s->%s->%s index: %zu overlap: %zu (%.0f%%)\n",
             word, kPhonArea, lex1_area, kPhonArea, phon_index, phon_overlap,
             phon_overlap * 100.0f / params_.phon_k);
    }
    if (phon_overlap > 0.7f * params_.phon_k) {
      return false;
    }
  }

  SetLogLevel(prev_log_level);
  return true;
}

bool LearnerBrain::TestAllWords(bool use_context, bool test_cross_input,
                                int log_level) {
  for (uint32_t w = 0; w < lex_size_; ++w) {
    if (!TestWord(w, use_context, test_cross_input, log_level)) {
      return false;
    }
  }
  return true;
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

void LearnerBrain::AnalyseAssemblies(const std::string& from,
                                     const std::string& to,
                                     size_t start, size_t end) {
  const Area& area_to = GetArea(to);
  std::vector<std::set<uint32_t>> all_assemblies;
  std::map<uint32_t, uint32_t> mult;
  float total_w = 0;
  size_t num_synapses = 0;
  size_t num_sat_weights = 0;
  for (uint32_t index = start; index < end; ++index) {
    ActivateArea(from, index);
    Project({{from, {to}}}, 1, false);
    AnalyseInput(from, to, total_w, num_synapses, num_sat_weights);
    const auto& activated = area_to.activated;
    std::set<uint32_t> assembly(activated.begin(), activated.end());
    all_assemblies.emplace_back(std::move(assembly));
    for (auto n : activated) {
      ++mult[n];
    }
  }
  printf("%s -> %s total input: %f num synapses: %zu num saturated: %zu\n",
         from.c_str(), to.c_str(), total_w, num_synapses, num_sat_weights);
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

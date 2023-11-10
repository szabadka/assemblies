cc_library(
    name = "brain",
    srcs = ["brain.cc"],
    hdrs = [
        "brain.h",
        "brain_util.h",
    ],
    deps = [],
)

cc_library(
    name = "learner",
    srcs = ["learner.cc"],
    hdrs = ["learner.h"],
    deps = [":brain"],
)

cc_test(
    name = "brain_test",
    srcs = ["brain_test.cc"],
    deps = [
        ":brain",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "learner_test",
    srcs = ["learner_test.cc"],
    deps = [
        ":learner",
        "@com_google_googletest//:gtest_main",
    ],
)

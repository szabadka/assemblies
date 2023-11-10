#! /usr/bin/python

import learner

def main():
    learner_brain = learner.LearnBrain(0.05, LEX_n=100000, num_nouns=2, num_verbs=2)
    learner_brain.train_experiment()
    learner_brain.test_word("DOG")


if __name__ == '__main__':
  main()

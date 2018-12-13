package algorithms.mdp;

import java.util.HashMap;
import java.util.Map.Entry;
import learning.*;
import problems.maze.*;

/**
 * Implements the value iteration algorithm for Markov Decision Processes
 */
public class ValueIteration extends LearningAlgorithm {

    /**
     * Stores the utilities for each state
     */
    private HashMap<State, Double> utilities;

    /**
     * Max delta. Controls convergence.
     */
    private double maxDelta = 0.01;

    /**
     * Learns the policy (notice that this method is protected, and called from the public method learnPolicy(LearningProblem problem, double gamma) in LearningAlgorithm.
     */
    @Override
    protected void learnPolicy() {
        // This algorithm only works for MDPs
        if (!(problem instanceof MDPLearningProblem)) {
            System.out.println("The algorithm ValueIteration can not be applied to this problem (model is not visible).");
            System.exit(0);
        }

        /* Used variables */
        MazeProblemMDP problemMDP = (MazeProblemMDP) this.problem; // Instance of the problem casted as MazeProblemMDP to work easier
        utilities = new HashMap<State, Double>(); // Initialize the HashMap of utilities
        HashMap<State, Double> currentUtilities; // Used to store the calculated utilities for the current iteration
        double delta = 0;

        /* Iterates through all the posible states,.. */
        for (State state : problemMDP.getAllStates()) {
            if (!problemMDP.isFinal(state)) {
                utilities.put(state, (double) 0); // assigning 0 to all the non final states
            } else {
                utilities.put(state, problemMDP.getReward(state)); // or the corresponding reward -100/100 in case of a final state
            }
        }

        /* Iterates until the delta converges to the set delta */
        while (!(delta < maxDelta * (1 - problemMDP.gamma) / problem.gamma)) { // Until delta < (1 - gamma)/gamma
            delta = 0; // Initializes delta
            currentUtilities = new HashMap<State, Double>();
            for (State state : problemMDP.getAllStates()) { // For each state among all possible states
                if (!problemMDP.isFinal(state)) { // If it is not a final state
                    double expectedUtility = Double.NEGATIVE_INFINITY;

                    /* Calculates for each possible action the expected utility */
                    for (Action action : problemMDP.getPossibleActions(state)) {
                        if (problemMDP.getExpectedUtility(state, action, utilities, problemMDP.gamma) > expectedUtility) {
                            expectedUtility = problemMDP.getExpectedUtility(state, action, utilities, problemMDP.gamma);
                        }
                    }

                    /* Gets the final utility and action for that state in the current iteration */
                    double newUtility = problemMDP.getReward(state) + problemMDP.gamma * expectedUtility;
                    currentUtilities.put(state, newUtility);

                    /* Updates the value of delta */
                    if (Math.abs(newUtility - utilities.get(state)) > delta) {
                        delta = Math.abs(newUtility - utilities.get(state));
                    }
                } else {
                    currentUtilities.put(state, problemMDP.getReward(state)); // For final states, the utility is the reward of 
                }
            }
            /* Updates policies U <-- U' */
            utilities = currentUtilities;
        }

        /* Obtains the optimal policy for each state */
        for (State state : problemMDP.getAllStates()) {
            if (!problemMDP.isFinal(state)) {
                Action optimalAction = null;
                double expectedUtility = Double.NEGATIVE_INFINITY;
                
                /* For each possible action, finds the one that leads to a higher utility */
                for (Action action : problemMDP.getPossibleActions(state)) {
                    if (problemMDP.getExpectedUtility(state, action, utilities, problemMDP.gamma) > expectedUtility) {
                        expectedUtility = problemMDP.getExpectedUtility(state, action, utilities, problemMDP.gamma);
                        optimalAction = action;
                    }
                }

                solution.setAction(state, optimalAction);
            }
        }
    }

    /**
     * Sets the parameters of the algorithm.
     */
    @Override
    public void setParams(String[] args) {
        // In this case, there is only one parameter (maxDelta).
        if (args.length > 0) {
            try {
                maxDelta = Double.parseDouble(args[0]);
            } catch (Exception e) {
                System.out.println("The value for maxDelta is not correct. Using 0.01.");
            }
        }
    }

    /**
     * Prints the results
     */
    public void printResults() {
        // Prints the utilities.
        System.out.println("Value Iteration\n");
        System.out.println("Utilities");
        for (Entry<State, Double> entry : utilities.entrySet()) {
            State state = entry.getKey();
            double utility = entry.getValue();
            System.out.println("\t" + state + "  ---> " + utility);
        }
        // Prints the policy
        System.out.println("\nOptimal policy");
        System.out.println(solution);
    }

    /**
     * Main function. Allows testing the algorithm with MDPExProblem
     */
    public static void main(String[] args) {
        LearningProblem mdp = new problems.mdpexample2.MDPExProblem();
        mdp.setParams(null);
        ValueIteration vi = new ValueIteration();
        vi.setProblem(mdp);
        vi.learnPolicy(mdp);
        vi.printResults();

    }

}

package algorithms.mdp;

import java.util.HashMap;

import learning.*;
import problems.maze.MazeProblemMDP;

public class PolicyIteration extends LearningAlgorithm {

    /**
     * Max delta. Controls convergence.
     */
    private double maxDelta = 0.01;

    /**
     * Learns the policy (notice that this method is protected, and called from the public method learnPolicy(LearningProblem problem, double gamma) in LearningAlgorithm.
     */
    @Override
    protected void learnPolicy() {
        if (!(problem instanceof MDPLearningProblem)) {
            System.out.println("The algorithm PolicyIteration can not be applied to this problem (model is not visible).");
            System.exit(0);
        }

        // Initializes the policy randomly
        solution = new Policy();
        Policy policyAux = new Policy();

        MazeProblemMDP problemMDP = (MazeProblemMDP) this.problem;
        HashMap<State, Double> utilities;

        /* Sets a random policy for each non-final state */
        for (State state : problemMDP.getAllStates()) {
            if (!problemMDP.isFinal(state)) {
                policyAux.setAction(state, problemMDP.randomAction(state));
            }
        }

        // Main loop of the policy iteration.
        
        /* While the new policy is not the same as the previous policy, iterate */
        while (!solution.equals(policyAux)) {
            solution = policyAux; 
            utilities = policyEvaluation(solution); 
            policyAux = policyImprovement(utilities); 
        }
    }

    /**
     * Policy evaluation. Calculates the utility given the policy
     */
    private HashMap<State, Double> policyEvaluation(Policy policy) {

        // Initializes utilities. In case of terminal states, the utility corresponds to
        // the reward. In the remaining (most) states, utilities are zero.		
        HashMap<State, Double> utilities = new HashMap<State, Double>();
        MDPLearningProblem problemMDP = (MDPLearningProblem) this.problem;
        double delta = 0;

        /* Iterates through all the posible states,.. */
        for (State state : problemMDP.getAllStates()) {
            if (!problemMDP.isFinal(state)) {
                utilities.put(state, (double) 0); // assigning 0 to all the non final states
            } else {
                utilities.put(state, problemMDP.getReward(state)); // or the corresponding reward -100/100 in case of a final state
            }
        }

        
        while (!(delta < maxDelta * (1 - problemMDP.gamma) / problem.gamma)) {
            delta = 0;
            for (State state : problemMDP.getAllStates()) {
                if (!problemMDP.isFinal(state)) {
                    
                    /* Calculated the expected utilty */
                    double expectedUtility = problemMDP.getExpectedUtility(state, policy.getAction(state), utilities, problemMDP.gamma);
                    
                    /* Obtains the new utility */
                    double newUtility = problemMDP.getReward(state) + problemMDP.gamma * expectedUtility;
                    utilities.put(state, newUtility); // Introduces it into the set of new utilities

                    /* Updates delta */
                    if (Math.abs(newUtility - utilities.get(state)) > delta) {
                        delta = Math.abs(newUtility - utilities.get(state));
                    }
                }
            }
        }

        return utilities;
    }

    /**
     * Improves the policy given the utility
     */
    private Policy policyImprovement(HashMap<State, Double> utilities) {
        // Creates the new policy
        Policy newPolicy = new Policy();
        MDPLearningProblem problemMDP = (MDPLearningProblem) this.problem;

        double expectedUtility = Double.NEGATIVE_INFINITY;

        /* Iterates through each state to find their optimal policies */
        for (State state : problemMDP.getAllStates()) {
            if (!problemMDP.isFinal(state)) {
                Action optimalAction = null;

                for (Action action : problemMDP.getPossibleActions(state)) {
                    if (problemMDP.getExpectedUtility(state, action, utilities, problemMDP.gamma) > expectedUtility) {
                        expectedUtility = problemMDP.getExpectedUtility(state, action, utilities, problemMDP.gamma);
                        optimalAction = action;
                    }
                }

                newPolicy.setAction(state, optimalAction);
            }
        }

        return newPolicy;
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
        System.out.println("Policy Iteration");
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
        PolicyIteration pi = new PolicyIteration();
        pi.setProblem(mdp);
        pi.learnPolicy(mdp);
        pi.printResults();
    }

}

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.lang.Math;
import java.util.Scanner;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.*;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.boaframework.SortedOutcomeSpace;

class Logistic {

    /**
     * the learning rate
     */
    private double rate;

    /**
     * the weight to learn
     */
    private double[] weights;

    /**
     * the number of iterations
     */
    private int ITERATIONS = 600;

    public Logistic(int n) {
        this.rate = 0.001;
        weights = new double[n];
        // initialize weights to random negative values
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() * -10;
        }
    }

    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public void train(List<Instance> instances) {
        for (int n = 0; n < ITERATIONS; n++) {
            double lik = 0.0;
            for (int i = 0; i < instances.size(); i++) {
                int[] x = instances.get(i).x;
                double predicted = classify(x);
                int label = instances.get(i).label;
                for (int j = 0; j < weights.length; j++) {
                    weights[j] = weights[j] + rate * (label - predicted) * x[j];
                }
                // not necessary for learning
                lik += label * Math.log(classify(x)) + (1 - label) * Math.log(1 - classify(x));
            }
//            System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
        }
    }

    double classify(int[] x) {
        double logit = .0;
        for (int i = 0; i < weights.length; i++) {
            logit += weights[i] * x[i];
        }
        return sigmoid(logit);
    }

    public static class Instance {
        public int label;
        public int[] x;

        public Instance(int label, int[] x) {
            this.label = label;
            this.x = x;
        }
    }

    public static List<Instance> readDataSet(String file) throws FileNotFoundException {
        List<Instance> dataset = new ArrayList<Instance>();
        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(file));
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (line.startsWith("#")) {
                    continue;
                }
                String[] columns = line.split("\\s+");

                // skip first column and last column is the label
                int i = 1;
                int[] data = new int[columns.length - 2];
                for (i = 1; i < columns.length - 1; i++) {
                    data[i - 1] = Integer.parseInt(columns[i]);
                }
                int label = Integer.parseInt(columns[i]);
                Instance instance = new Instance(label, data);
                dataset.add(instance);
            }
        } finally {
            if (scanner != null)
                scanner.close();
        }
        return dataset;
    }

    public static void main(String... args) throws FileNotFoundException {
        List<Instance> instances = readDataSet("dataset.txt");
        Logistic logistic = new Logistic(5);
        logistic.train(instances);
        int[] x = {2, 1, 1, 0, 1};
        System.out.println("prob(1|x) = " + logistic.classify(x));

        int[] x2 = {1, 0, 1, 0, 0};
        System.out.println("prob(1|x2) = " + logistic.classify(x2));

    }

}

public class Team8 extends AbstractNegotiationParty {
    private Bid lastOffer;
    // Stores the last offer that was proposed
    // by the opponent.
    private Bid constantBid;
    // Stores the best possible bid we can have, and then
    // uses this as the standard to compare with the proposed
    // bids.
    private double reservationValue;
    // Utility Threshold is the utility we get by not accepting
    // at any point. This is possible since we k=now the random
    // agent stops after 90% of the deadline passes.
    private int numRounds = 0;
    // private HashMap<Bid, Integer> proposedBids = new HashMap<>();

    // --List of regressions, one corresponding to each of the opponent.
//    private Logistic logReg;
    // --A map from all the possible agents (Agent IDs) to the index in the Logistic class.
    private HashMap<AgentID, Logistic> OpponentMaps = new HashMap<>();

    private HashMap<AgentID, List<Logistic.Instance>> OpponentInstances = new HashMap<>();
    private double individualUtilThreshold = 1, individualUtilThresholdDelta = 0.02;

    private Bid lastBidOnTable = null;
    int totalValues = 0;

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);

        SortedOutcomeSpace outcomeSpace = new SortedOutcomeSpace(info.getUtilitySpace());
        constantBid = outcomeSpace.getMaxBidPossible().getBid();
        // This stores the constant bid that
        // our agent is going to propose.

        double totalRun = info.getDeadline().getValue();
        double expectedRun = 0.9 * totalRun;
        
        reservationValue = info.getUtilitySpace().getReservationValueUndiscounted();
        individualUtilThresholdDelta = (1 - reservationValue * info.getUtilitySpace().getDiscountFactor())/60;
        // Now utility threshold contains the minimum bid our agent gets
        // as long as there isn't any disagreement.

        List<Issue> allIssues = utilitySpace.getDomain().getIssues();

        for (Issue rawIssue : allIssues) {
            IssueDiscrete issue = (IssueDiscrete) rawIssue;
            totalValues += issue.getNumberOfValues();
        }

//        logReg = new Logistic(totalValues);
    }

    /*
     * Calculate the score of a bid
     * beta * social_score(bid) + (1 - beta) * individual_score(bid)
     * TODO: Update individualUtilThreshold over rounds
     */
    public double getBidScore(Bid bid) {
        double beta = 0.6;

        // social score is our utility of the bid + square of probability of opponent acceptance (as per logreg)
//        double opponentProb = this.logReg.classify(oneHotEncoder(bid));
//        double socialScore = (getUtilityWithDiscount(bid) + Math.pow(opponentProb,2) )/ 2;
//
//        double individualScore = (getUtilityWithDiscount(bid) < individualUtilThreshold) ? 0 : getUtilityWithDiscount(bid);
//
//        if ( opponentProb >= getUtility(bid) ) {
//            return 0;
//        }
//        return beta * socialScore + (1 - beta) * individualScore;

        double allOpponentProbSum = 0;
        for (AgentID agentID : this.OpponentInstances.keySet()) {
            System.out.println("Agent ID: " + agentID);
            Logistic logistic = this.OpponentMaps.get(agentID);
//            System.out.println("One-Hot " + oneHotEncoder(bid).toString() );
            System.out.println(logistic == null);
            double opponentProb = logistic.classify(oneHotEncoder(bid));
            allOpponentProbSum += Math.pow(opponentProb,2);
        }

//        allOpponentProb /= OpponentInstances.size();
        double socialScore = (getUtilityWithDiscount(bid) + allOpponentProbSum)/ ( OpponentInstances.size() + 1 );
        double individualScore = (getUtilityWithDiscount(bid) < individualUtilThreshold) ? 0 : getUtilityWithDiscount(bid);

//        if ( allOpponentProbSum >= getUtility(bid) ) {
//            return 0;
//        }
        return beta * socialScore + (1 - beta) * individualScore;

    }
    
    /*
     * Iterate through all bids with utility higher than the reservation value
     * and return the one with the highest score using our scoring function and.
     */
    public Bid findBestBid() {
        SortedOutcomeSpace outcomeSpace = new SortedOutcomeSpace(utilitySpace);
        Bid bestBid = null;
        double bestScore = 0.0;

        for (BidDetails bidDetails : outcomeSpace.getAllOutcomes()) {
            Bid bid = bidDetails.getBid();
            double utility = getUtility(bid);
           if (utility < reservationValue) {
                continue;
            }
            double score = getBidScore(bid);
            if (score > bestScore) {
                bestBid = bid;
                bestScore = score;
            }
        }

        return bestBid;
    }

    /*
     * Checks if the proposed bid has utility greater than
     * what we eventually will have at the end, and accepts
     * if that is the case.
     */
    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions) {
        for (AgentID agentID : this.OpponentInstances.keySet()) {
            Logistic logistic = new Logistic(totalValues);
            logistic.train(this.OpponentInstances.get(agentID));
            this.OpponentMaps.put(agentID, logistic);
        }

        if (lastOffer != null) {
            System.out.println("Last offer: " + lastOffer);
            System.out.println("Last offer score: " + getBidScore(lastOffer));
            System.out.println("Best bid: " + findBestBid());
            System.out.println("Best bid score: " + getBidScore(findBestBid()));
            
            // Need to check if 40 % of the rounds have elapsed.
            // If not, we do not accept the bid.
            
            
            if ( (  getBidScore(lastOffer) >= getBidScore(findBestBid()) )
                    && ( (double)getUtility(lastOffer) > (double)utilitySpace.getReservationValue())
                    || ( timeline.getTime() >= 0.96 ) ) {
                return new Accept(getPartyId(), lastOffer);
                // If the opponent proposes a bid that
                // is greater than what we are eventually going to get,
                // we accept.
            }
        }

//        logReg = new Logistic(totalValues);
//        logReg.train(this.instances);

        // call a function that will iterate over all possible bids
        // and then select one with maximum score.

//        instances.add(new Logistic.Instance(0, oneHotEncoder(findBestBid())));
        individualUtilThreshold = Math.max(individualUtilThreshold - individualUtilThresholdDelta, utilitySpace.getReservationValue());

        Bid ourBestBid = findBestBid();
        this.lastBidOnTable = ourBestBid;
        return new Offer(getPartyId(), ourBestBid);
    }
    
    /*
     * Stores the received bid in a variable
     * */
    @Override
    public void receiveMessage(AgentID sender, Action action) {
        if (action instanceof Offer) {
            this.OpponentInstances.putIfAbsent(sender, new ArrayList<>());
//            this.numRounds += 1;

            // The last bid was a negative sample for this opponent.
            if ( this.lastBidOnTable != null ) {
                this.OpponentInstances.get(sender).add(new Logistic.Instance(0, oneHotEncoder(this.lastBidOnTable)));
            }

            lastOffer = ((Offer) action).getBid();
            this.lastBidOnTable = lastOffer;

            // Add the last offer to the list of instances
            this.OpponentInstances.get(sender).add(new Logistic.Instance(1, oneHotEncoder(lastOffer)));
        }

        if ( action instanceof Accept ) {
            this.numRounds += 1;
            this.OpponentInstances.get(sender).add(new Logistic.Instance(1, oneHotEncoder(this.lastBidOnTable)));
        }
    }

    /*
     * Refuses to elaborate further.
     * */
    @Override
    public String getDescription() {
        return "Refusing to elaborate further.";
    }

    public int[] oneHotEncoder(Bid thisBid) {
        List<Issue> allIssues = utilitySpace.getDomain().getIssues();

        HashMap<Integer, Value> valueMap = thisBid.getValues();
        List<Integer> encoding = new ArrayList<>();

        for (Issue rawIssue : allIssues) {
            IssueDiscrete issue = (IssueDiscrete) rawIssue;
            Integer issueID = issue.getNumber();

            int valueID = issue.getValueIndex((ValueDiscrete) valueMap.get(issueID));
            int numValues = issue.getNumberOfValues();

            for (int i = 0; i < numValues; i++) {
                if (i == valueID) {
                    encoding.add(1);
                } else {
                    encoding.add(0);
                }
            }
        }

        int[] x = new int[encoding.size()];

        for (int i = 0; i < encoding.size(); i++) {
            x[i] = encoding.get(i);
            System.out.print(x[i]);
        }
        System.out.println();

        return x;
    }
}

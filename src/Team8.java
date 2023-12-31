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
    private int ITERATIONS = 500;

    public Logistic(int n) {
        this.rate = 0.001;
        weights = new double[n];
        // initialize weights to random negative values
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() * -1;
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

    private Logistic logReg;

    private List<Logistic.Instance> instances = new ArrayList<Logistic.Instance>();

    private double individualUtilThreshold = 1, individualUtilThresholdDelta = 0.02;

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
        individualUtilThresholdDelta = (1 - reservationValue * info.getUtilitySpace().getDiscountFactor()) / 60;
        // Now utility threshold contains the minimum bid our agent gets
        // as long as there isn't any disagreement.

        List<Issue> allIssues = utilitySpace.getDomain().getIssues();

        for (Issue rawIssue : allIssues) {
            IssueDiscrete issue = (IssueDiscrete) rawIssue;
            totalValues += issue.getNumberOfValues();
        }

        logReg = new Logistic(totalValues);
    }

    /*
     * Calculate the score of a bid
     * beta * social_score(bid) + (1 - beta) * individual_score(bid)
     * TODO: Update individualUtilThreshold over rounds
     */
    public double getBidScore(Bid bid, boolean verbose) {
        double beta = 0.5;

        // social score is our utility of the bid + square of probability of opponent acceptance (as per logreg)
        double opponentProb = this.logReg.classify(oneHotEncoder(bid));
        double ourDiscountedUtil = getUtilityWithDiscount(bid);
        double socialScore = (ourDiscountedUtil + Math.pow(opponentProb, 2)) / 2;

        double individualScore = (ourDiscountedUtil < individualUtilThreshold) ? 0 : opponentProb;
        if (verbose) {
            System.out.println("Bid: " + bid);
            System.out.println("Social score: " + socialScore + " Individual score: " + individualScore);
            System.out.println("Opponent prob: " + opponentProb);
        }
        if  (opponentProb >= getUtility(bid)) {
            return 0;
        }

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
            double score = getBidScore(bid, false);
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
        if (lastOffer != null) {
            System.out.println("-------------------------------------");
            System.out.println("Last offer: " + lastOffer);
            double lastBidScore = getBidScore(lastOffer, true);
            System.out.println("-------------------------------------");
            System.out.println("Best bid: " + findBestBid());
            double bestBidScore = getBidScore(findBestBid(), true);
            if (lastBidScore >= bestBidScore && getUtilityWithDiscount(lastOffer) >= utilitySpace.getReservationValue()) {
                return new Accept(getPartyId(), lastOffer);
            }
        }

        logReg = new Logistic(totalValues);
        logReg.train(this.instances);

        // call a function that will iterate over all possible bids
        // and then select one with maximum score.

        instances.add(new Logistic.Instance(0, oneHotEncoder(findBestBid())));
        individualUtilThreshold = Math.max(individualUtilThreshold - individualUtilThresholdDelta, utilitySpace.getReservationValue());
        return new Offer(getPartyId(), findBestBid());
    }

    /*
     * Stores the received bid in a variable
     * */
    @Override
    public void receiveMessage(AgentID sender, Action action) {
        if (action instanceof Offer) {
            lastOffer = ((Offer) action).getBid();
            this.instances.add(new Logistic.Instance(1, oneHotEncoder(lastOffer)));
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

//        System.out.println("-------------------------------------");
//        for (int val : encoding) {
//            System.out.print(val);
//        }

        int[] x = new int[encoding.size()];

        for (int i = 0; i < encoding.size(); i++) {
            x[i] = encoding.get(i);
        }

        return x;
    }
}
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.lang.Math;
import java.util.Scanner;

import genius.core.Agent;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Inform;
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
    private final double rate;

    /**
     * the weight to learn
     */
    private final double[] weights;

    /**
     * the number of iterations
     */
    private final int ITERATIONS = 600;

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
    private double reservationValue;

    private final HashMap<AgentID, Logistic> OpponentMaps = new HashMap<>();

    private final HashMap<AgentID, List<Logistic.Instance>> OpponentInstances = new HashMap<>();

    private final HashMap<AgentID, HashMap<Bid, Integer>> OpponentPositiveBidCount = new HashMap<>();
    private final HashMap<AgentID, HashMap<Bid, Integer>> OpponentNegativeBidCount = new HashMap<>();

    private double individualUtilThreshold = 1, individualUtilThresholdDelta = 0.02;

    private Bid lastBidOnTable = null;
    int totalValues = 0;

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);

        SortedOutcomeSpace outcomeSpace = new SortedOutcomeSpace(info.getUtilitySpace());
        Bid constantBid = outcomeSpace.getMaxBidPossible().getBid();
        // This stores the constant bid that
        // our agent is going to propose.

        double totalRun = info.getDeadline().getValue();
        double expectedRun = 0.9 * totalRun;
        
        reservationValue = info.getUtilitySpace().getReservationValueUndiscounted();
        individualUtilThresholdDelta = (1 - 0.75 * info.getUtilitySpace().getDiscountFactor())/60;
        // Now utility threshold contains the minimum bid our agent gets
        // as long as there isn't any disagreement.

        List<Issue> allIssues = utilitySpace.getDomain().getIssues();

        for (Issue rawIssue : allIssues) {
            IssueDiscrete issue = (IssueDiscrete) rawIssue;
            totalValues += issue.getNumberOfValues();
        }

    }

    /*
     * Calculate the score of a bid
     * beta * social_score(bid) + (1 - beta) * individual_score(bid)
     * TODO: Update individualUtilThreshold over rounds
     */
    public double getBidScore(Bid bid) {
        double beta = 0.6;

        double allOpponentProbSum = 0, allOpponentProbSumSquare = 0;

        for (AgentID agentID : this.OpponentInstances.keySet()) {
            Logistic logistic = this.OpponentMaps.get(agentID);

            if ( logistic == null ){
                System.out.println("Regression is null");
            }

            double opponentProb = logistic.classify(oneHotEncoder(bid));
            allOpponentProbSum += opponentProb;
            allOpponentProbSumSquare += Math.pow(opponentProb,2);
        }

        if ( OpponentInstances.size() != 0 ) {
            allOpponentProbSum /= (OpponentInstances.size());
        }

        double socialScore = (getUtilityWithDiscount(bid) + allOpponentProbSumSquare)/ ( OpponentInstances.size() + 1 );
        double individualScore = (getUtilityWithDiscount(bid) < individualUtilThreshold) ? 0 : allOpponentProbSum;

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


    private void updateInstanceList( AgentID agent ){
        HashMap<Bid, Integer> agentPositiveHashMap = this.OpponentPositiveBidCount.get(agent);

        if ( agentPositiveHashMap != null ) {
            for (Bid opponentSampleBid : agentPositiveHashMap.keySet()) {
                int bidCount = agentPositiveHashMap.get(opponentSampleBid);
                int desiredFrequency = (int) (bidCount );

                for (int i = 0; i < desiredFrequency; i++) {
                    this.OpponentInstances.get(agent).add(new Logistic.Instance(1, oneHotEncoder(opponentSampleBid)));
                }
            }
        }

        HashMap<Bid, Integer> agentNegativeHashMap = this.OpponentNegativeBidCount.get(agent);

        if ( agentNegativeHashMap != null ) {
            for (Bid opponentSampleBid : agentNegativeHashMap.keySet()) {
                int bidCount = agentNegativeHashMap.get(opponentSampleBid);
                int desiredFrequency = (int) (bidCount );

                for (int i = 0; i < desiredFrequency; i++) {
                    this.OpponentInstances.get(agent).add(new Logistic.Instance(0, oneHotEncoder(opponentSampleBid)));
                }
            }
        }
    }

    private void updateInstances() {
        for (AgentID agentID : this.OpponentInstances.keySet()) {
            updateInstanceList(agentID);
        }
    }

    /*
     * Checks if the proposed bid has utility greater than
     * what we eventually will have at the end, and accepts
     * if that is the case.
     */
    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions) {
        // Update the instances for each of the opponent.
        updateInstances();

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
            
            if ( (  getUtilityWithDiscount(lastOffer) >= getUtilityWithDiscount(findBestBid()) )
                    && ( getUtility(lastOffer) > utilitySpace.getReservationValue())
                    || ( timeline.getTime() >= 0.96 ) ) {
                return new Accept(getPartyId(), lastOffer);
                // If the opponent proposes a bid that
                // is greater than what we are eventually going to get,
                // we accept.
                // The last condition ensures that we accept the last bid.
            }
        }

        individualUtilThreshold = Math.max(individualUtilThreshold - individualUtilThresholdDelta, utilitySpace.getReservationValue());

        Bid ourBestBid = findBestBid();
        this.lastBidOnTable = ourBestBid;
        return new Offer(getPartyId(), ourBestBid);
    }
    
    /*
     * Stores the received bids and updates
     * the received bid instances for the sender agent.
     * */
    @Override
    public void receiveMessage(AgentID sender, Action action) {
        if (action instanceof Offer) {
            // Add this agent to the keys in OpponentInstances if it is not already there.
            this.OpponentInstances.putIfAbsent(sender, new ArrayList<>());

            // First thing to do is to add this sample to the opponent's positive bid count.
            this.OpponentPositiveBidCount.putIfAbsent(sender, new HashMap<>());
            this.OpponentPositiveBidCount.get(sender).putIfAbsent( ((Offer)action).getBid(), 0);
            this.OpponentPositiveBidCount.get(sender).put( ((Offer)action).getBid(), 1 + this.OpponentPositiveBidCount.get(sender).get(((Offer)action).getBid()));

            // The last bid was a negative sample for this opponent.
            if ( this.lastBidOnTable != null ) {
                this.OpponentNegativeBidCount.putIfAbsent(sender, new HashMap<>());
                this.OpponentNegativeBidCount.get(sender).putIfAbsent( this.lastBidOnTable, 0 );
                this.OpponentNegativeBidCount.get(sender).put( this.lastBidOnTable, 1 + this.OpponentNegativeBidCount.get(sender).get(this.lastBidOnTable));
            }

            lastOffer = ((Offer) action).getBid();
            this.lastBidOnTable = lastOffer;
        }

        // The last bid was a positive sample for this opponent.
        if ( action instanceof Accept ) {
            this.OpponentPositiveBidCount.putIfAbsent(sender, new HashMap<>());
            this.OpponentPositiveBidCount.get(sender).putIfAbsent( this.lastBidOnTable, 0);
            this.OpponentPositiveBidCount.get(sender).put( this.lastBidOnTable, 1 + this.OpponentPositiveBidCount.get(sender).get(this.lastBidOnTable) );
        }
    }

    /*
     * Refuses to elaborate further.
     * */
    @Override
    public String getDescription() {
        return "Linear-Linear";
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
        }

        return x;
    }
}

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
    private int ITERATIONS = 3000;

    public Logistic(int n) {
        this.rate = 0.0001;
        weights = new double[n];
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
            System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
        }
    }

    private double classify(int[] x) {
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
    private double utilityThreshold;
    // Utility Threshold is the utility we get by not accepting
    // at any point. This is possible since we k=now the random
    // agent stops after 90% of the deadline passes.

    private Logistic logReg;

    private List<Logistic.Instance> instances = new ArrayList<Logistic.Instance>();

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);

        SortedOutcomeSpace outcomeSpace = new SortedOutcomeSpace(info.getUtilitySpace());
        constantBid = outcomeSpace.getMaxBidPossible().getBid();
        // This stores the constant bid that
        // our agent is going to propose.

        double totalRun = info.getDeadline().getValue();
        double expectedRun = 0.9 * totalRun;

        utilityThreshold = Math.pow(utilitySpace.getDiscountFactor(), expectedRun / totalRun);
        utilityThreshold = utilityThreshold * getUtility(constantBid);
        // Now utility threshold contains the minimum bid our agent gets
        // as long as there isn't any disagreement.

        int totalValues = 0;
        List<Issue> allIssues = utilitySpace.getDomain().getIssues();

        for (Issue rawIssue : allIssues) {
            IssueDiscrete issue = (IssueDiscrete) rawIssue;
            totalValues += issue.getNumberOfValues();
        }

        this.logReg = new Logistic(totalValues);

    }

    /*
     * Checks if the proposed bid has utility greater than
     * what we eventually will have at the end, and accepts
     * if that is the case.
     */
    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions) {
        if (lastOffer != null) {
            if (getUtilityWithDiscount(lastOffer) >= utilityThreshold) {
                return new Accept(getPartyId(), lastOffer);
                // If the opponent proposes a bid that
                // is greater than what we are eventually going to get,
                // we accept.
            }
        }

        this.logReg.train(this.instances);

        // call a function that will iterate over all possible bids
        // and then select one with maximum score.

        this.instances.add(new Logistic.Instance(0, oneHotEncoder(constantBid)));
        return new Offer(getPartyId(), constantBid);
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
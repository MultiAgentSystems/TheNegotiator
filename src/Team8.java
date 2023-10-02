import java.util.List;
import java.lang.Math;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.boaframework.SortedOutcomeSpace;


private class
public class Team8 extends AbstractNegotiationParty
{
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

    @Override
    public void init(NegotiationInfo info)
    {
        super.init(info);

        SortedOutcomeSpace outcomeSpace = new SortedOutcomeSpace(info.getUtilitySpace());
        constantBid = outcomeSpace.getMaxBidPossible().getBid();
        // This stores the constant bid that
        // our agent is going to propose.

        double totalRun = info.getDeadline().getValue();
        double expectedRun = 0.9*totalRun;

        utilityThreshold = Math.pow(utilitySpace.getDiscountFactor(), expectedRun/totalRun);
        utilityThreshold = utilityThreshold * getUtility(constantBid);
        // Now utility threshold contains the minimum bid our agent gets
        // as long as there isn't any disagreement.
    }

    /*
    * Checks if the proposed bid has utility greater than
    * what we eventually will have at the end, and accepts
    * if that is the case.
    */
    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions)
    {
        if ( lastOffer != null) {
            if (getUtilityWithDiscount(lastOffer) >= utilityThreshold) {
                return new Accept(getPartyId(), lastOffer);
                // If the opponent proposes a bid that
                // is greater than what we are eventually going to get,
                // we accept.
            }
        }

        return new Offer(getPartyId(),constantBid );
    }

    /*
    * Stores the received bid in a variable
    * */
    @Override
    public void receiveMessage(AgentID sender, Action action)
    {
        if (action instanceof Offer)
        {
            lastOffer = ((Offer) action).getBid();
        }
    }

    /*
    * Refuses to elaborate further.
    * */
    @Override
    public String getDescription()
    {
        return "Refusing to elaborate further.";
    }

}

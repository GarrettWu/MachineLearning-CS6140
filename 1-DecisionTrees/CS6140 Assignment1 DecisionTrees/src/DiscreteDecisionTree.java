import java.util.List;

/**
 * Decision tree for classification discrete data set, mushroom
 * @author Garrett
 *
 */
public class DiscreteDecisionTree {
	/**
	 * Root of the tree.
	 */
	public DiscreteTreeNode root;
	
	/**
	 * Minimum size to stop split.
	 */
	public int minSize;
	
	/**
	 * Constructor
	 * @param data
	 * @param eta
	 */
	DiscreteDecisionTree(List<DiscreteData> data, double eta) {
		this.minSize = (int)(data.size() * eta);
		this.root = new DiscreteTreeNode(data, this);
		
	}
	
	/**
	 * Predict to outcome by the tree.
	 * @param dd
	 * @return
	 */
	int predict(DiscreteData dd) {
		DiscreteTreeNode node = root;
		while (!node.isLeaf()) {
			int feature = node.split.feature;
			int index = dd.x[feature];
			node = node.children.get(index);
		}
		
		return node.getVote();
	}
}

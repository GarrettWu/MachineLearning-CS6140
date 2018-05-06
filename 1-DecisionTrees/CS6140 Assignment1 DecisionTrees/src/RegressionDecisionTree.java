import java.util.List;

/**
 * Regression decision tree for housing
 * @author Garrett
 *
 */
public class RegressionDecisionTree {
	/**
	 * Root of the tree.
	 */
	public RegressionTreeNode root;
	
	/**
	 * Minimum size to stop split.
	 */
	public int minSize;
	
	/**
	 * Constructor
	 * @param data
	 * @param eta
	 */
	RegressionDecisionTree(List<RegressionData> data, double eta) {
		this.minSize = (int)(data.size() * eta);
		this.root = new RegressionTreeNode(data, this);
		
	}
	
	/**
	 * Predict to outcome by the tree.
	 * @param data
	 * @return
	 */
	double predict(RegressionData rd) {
		RegressionTreeNode node = root;
		while (!node.isLeaf()) {
			if (rd.x[node.split.feature] <= node.split.value) {
				node = node.left;
			} else {
				node = node.right;
			}
		}
		
		return node.predictedValue();
	}
}

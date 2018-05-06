import java.util.List;

/**
 * Decision Tree to deal with classification continuous data
 * @author Garrett
 *
 */
public class ContinuousDecisionTree {
	/**
	 * Root of the tree.
	 */
	public ContinuousTreeNode root;
	
	/**
	 * Minimum size to stop split.
	 */
	public int minSize;
	
	/**
	 * Constructor
	 * @param irises
	 * @param eta
	 */
	ContinuousDecisionTree(List<ContinuousData> data, double eta) {
		this.minSize = (int)(data.size() * eta);
		this.root = new ContinuousTreeNode(data, this);
		
	}
	
	/**
	 * Predict to outcome by the tree.
	 * @param iris
	 * @return
	 */
	int predict(ContinuousData cd) {
		ContinuousTreeNode node = root;
		while (!node.isLeaf()) {
			if (cd.x[node.split.feature] <= node.split.value) {
				node = node.left;
			} else {
				node = node.right;
			}
		}
		
		return node.getVote();
	}
}

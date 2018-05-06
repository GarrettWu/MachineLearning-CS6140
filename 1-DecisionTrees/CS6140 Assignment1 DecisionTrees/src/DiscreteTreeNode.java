import java.util.ArrayList;
import java.util.List;

/**
 * Node of decision tree for discrete data, mushroom
 * @author Garrett
 *
 */
public class DiscreteTreeNode {
	/**
	 * The splitting feature and threshold of the node
	 * @author Garrett
	 *
	 */
	public static class Split {
		public int feature;

		Split(int feature) {
			this.feature = feature;
		}
	}
	/**
	 * List of data
	 */
	public List<DiscreteData> data;

	/**
	 * Split attribute and value of the node
	 */
	public Split split;

	/**
	 * Children nodes
	 */
	public List<DiscreteTreeNode> children;

	/**
	 * The tree of the nodes
	 */
	public DiscreteDecisionTree tree;

	/**
	 * Vote class (predict class)
	 */
	int vote = -1;
	
	/**
	 * Entropy
	 */
	double entropy = -1;


	/**
	 * Constructor
	 * @param data
	 * @param tree
	 */
	DiscreteTreeNode(List<DiscreteData> data, DiscreteDecisionTree tree) {
		this.data = data;	
		this.tree = tree;	
		this.split = getSplit();
	}

	/**
	 * Is a leaf node
	 * @return
	 */
	boolean isLeaf() {
		return split == null;
	}

	/**
	 * Size of the data in the node
	 * @return
	 */
	int size() {
		return data.size();
	}

	/**
	 * Get the entropy value
	 * @return
	 */
	double entropy() {
		if (entropy == -1) {
			entropy = Util.discreteEntropy(data);
		}
		return entropy;
	}

	/**
	 * Calculate the information gain of specific split
	 * @param split
	 * @return
	 */
	double informationGain(Split split) {
		double hq = entropy();
		int feature = split.feature;
		int size = data.get(0).getFeatureSize(feature);

		@SuppressWarnings("unchecked")
		List<DiscreteData>[] dataChildren = new List[size];
		for (int i = 0; i < dataChildren.length; i++) {
			dataChildren[i] = new ArrayList<DiscreteData>();
		}

		for (DiscreteData dd : data) {
			int v = dd.x[feature];
			dataChildren[v].add(dd);
		}

		for (List<DiscreteData> list : dataChildren) {
			double hc = Util.discreteEntropy(list) * list.size() / size();
			hq -= hc;
		}

		return hq;
	}

	/**
	 * Find the split maximize information gain
	 * @return
	 */
	public Split getSplit() {
		// if meet stop conditions
		if (entropy() == 0 || data.size() <= tree.minSize) {
			return null;
		}

		double maxIG = 0;
		Split maxSplit = null;

		// find max IG
		for (int f = 0; f < data.get(0).x.length; f++) {
			Split s = new Split(f);
			double ig = informationGain(s);

			if (ig > maxIG) {
				maxIG = ig;
				maxSplit = s;
			}
		}

		// split if possible
		if (maxSplit != null) {
			int size = data.get(0).getFeatureSize(maxSplit.feature);
			children = new ArrayList<DiscreteTreeNode>(size);

			@SuppressWarnings("unchecked")
			List<DiscreteData>[] dataChildren = new List[size];
			for (int i = 0; i < dataChildren.length; i++) {
				dataChildren[i] = new ArrayList<DiscreteData>();
			}

			for (DiscreteData i : data) {
				int v = i.x[maxSplit.feature];
				dataChildren[v].add(i);
			}

			for (int i = 0; i < dataChildren.length; i++) {
				children.add(new DiscreteTreeNode(dataChildren[i], tree));
			}
		}

		return maxSplit;
	}

	/**
	 * Get the predict value
	 * @return
	 */
	public int getVote() {
		if (vote != -1) {
			return vote;
		}

		int classes = DiscreteData.classes();
		int[] counts = new int[classes];
		for (DiscreteData dd : data) {
			counts[dd.y]++;
		}
		int max = 0, maxI = 0;
		for (int i = 0; i < classes; i++) {
			if (counts[i] > max) {
				maxI = i;
				max = counts[i];
			}
		}

		return maxI;
	}
}

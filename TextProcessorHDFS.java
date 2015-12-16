import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.lda.cvb.CVB0Driver;
import org.apache.mahout.text.SequenceFilesFromDirectory;
import org.apache.mahout.utils.vectors.RowIdJob;
import org.apache.mahout.utils.vectors.VectorDumper;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
/**
 * Hello world!
 *
 */

public class TextProcessorHDFS {

	private int textToSequenceFiles(Configuration conf, String input, String output) throws Exception {

		/*
		 * Pass the arguments to the run method in the same way we invoke Mahout
		 * from the terminal. This may not be the best way in comparison to
		 * calling Mahout API but I struggled enough and failed to find good
		 * resources/tutorials that describe the standard workflow to use Mahout
		 * API. All arguments are self explanatory, "--method sequential" asks
		 * Mahout not to run on Hadoop's MapReduce
		 * 
		 */
		String[] para = { "--input", input, "--output", output, "-c", "UTF-8", "--overwrite" };
		SequenceFilesFromDirectory x = new SequenceFilesFromDirectory();
		x.setConf(conf);
		int one = x.run(para);
		return one;
	}

	private int vectorize(Configuration conf, String input, String output) throws Exception {

		/*
		 * -wt tfidf --> Use the tfidf weighting method. -ng 2 --> Use an n-gram
		 * size of 2 to generate both unigrams and bigrams. -ml 50 --> Use a
		 * log-likelihood ratio (LLR) value of 50 to keep only very significant
		 * bigrams.
		 */

		String[] para = { "-o", output, "-i", input, "-ow", "-wt", "tf"};
		SparseVectorsFromSequenceFiles x = new SparseVectorsFromSequenceFiles();
		x.setConf(conf);
		int one = x.run(para);
		return one;
	}

	private int setrows(Configuration conf, String input, String output) throws Exception {

		/*
		 * convert sparse vectors to matrix form for cvb
		 */

		String[] para = { "-o", output, "-i", input};
		RowIdJob x = new RowIdJob();
		x.setConf(conf);
		int one = x.run(para);
		return one;
	}
	private int lda(Configuration conf, String input, String dictionary, String output) throws Exception {

		/*
		 * run cvb as lda
		 * -dict is the location of the dictionary file for vocab size
		 * -e is the topic smoothing parameter
		 * -maxIter is the max number of iterations
		 * -k is the number of topics
		 */

		String[] para = { "-o", output, "-i", input, "-dict", dictionary, "-k", "100", "-ow", "--maxIter", "100"};
		CVB0Driver x = new CVB0Driver();
		x.setConf(conf);
		int one = x.run(para);
		return one;
	}
	
	private void lda_topics(Configuration conf, String input, String dictionary, String output) throws Exception {

		/*
		 * get topics
		 * -dict is the location of the dictionary file for vocab size
		 * -e is the topic smoothing parameter
		 * -maxIter is the max number of iterations
		 * -k is the number of topics
		 */
		String filesystem_in = "file:/usr/local/hadoop/hadoop-2.7.1/hadoop_store/hdfs/datanode";
		filesystem_in.concat(input);
		String filesystem_out = "file:/usr/local/hadoop/hadoop-2.7.1/hadoop_store/hdfs/datanode";
		filesystem_out.concat(output);
		String[] para = { "-o", filesystem_out, "-i", filesystem_in, "-d", dictionary, "-dt", "sequencefile", "-vs", "10", "-sort", input};
		VectorDumper.main(para);
	}

	private void readDictionaryAndFrequency(String path1, String path2) throws IOException {
		Configuration conf = new Configuration();
		SequenceFile.Reader read1 = new SequenceFile.Reader(FileSystem.get(conf), new Path(path1), conf);
		SequenceFile.Reader read2 = new SequenceFile.Reader(FileSystem.get(conf), new Path(path2), conf);
		IntWritable dictionaryKey = new IntWritable();
		Text text = new Text();
		LongWritable freq = new LongWritable();
		HashMap<Integer, Long> freqMap = new HashMap<Integer, Long>();
		HashMap<Integer, String> dictionaryMap = new HashMap<Integer, String>();

		/*
		 * Read the contents of dictionary.file-0 and frequency.file-0 and write
		 * them to appropriate HashMaps
		 */

		while (read1.next(text, dictionaryKey)) {
			dictionaryMap.put(Integer.parseInt(dictionaryKey.toString()), text.toString());
		}
		while (read2.next(dictionaryKey, freq)) {
			freqMap.put(Integer.parseInt(dictionaryKey.toString()), Long.parseLong(freq.toString()));
		}

		read1.close();
		read2.close();

		for (int i = 0; i < dictionaryMap.size(); i++) {
			System.out.println("Key " + i + ": " + dictionaryMap.get(i));
		}

		for (int i = 0; i < freqMap.size(); i++) {
			System.out.println("Key " + i + ": " + freqMap.get(i));
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
//		conf.addResource("/usr/local/hadoop/hadoop-2.7.1/etc/hadoop/core-site.xml");
//		conf.addResource("/usr/local/hadoop/hadoop-2.7.1/etc/hadoop/hdfs-site.xml");
//		conf.set("fs.defaultFS", "hdfs://localhost:9000/");
		  System.setProperty("hadoop.home.dir", "c:\\apache-hadoop-2.7.1\\hadoop-2.7.1\\");
		  conf.set("fs.defaultFS", "file:///");
				
		TextProcessorHDFS t = new TextProcessorHDFS();
		t.textToSequenceFiles(conf, "/Courses/CSEE6893/HW2/wiki/articles_test", "/Courses/CSEE6893/HW2/wiki/yelp_seq");
		t.vectorize(conf, "/Courses/CSEE6893/HW2/wiki/yelp_seq", "/Courses/CSEE6893/HW2/wiki/yelp_vec");
		t.setrows(conf, "/Courses/CSEE6893/HW2/wiki/yelp_vec/tfidf-vectors", "/Courses/CSEE6893/HW2/wiki/yelp_mat/");
		t.lda(conf, "/Courses/CSEE6893/HW2/wiki/yelp_matrix/matrix", "/Courses/CSEE6893/HW2/wiki/yelp_vec/dictionary.file-0","/user/kyle/yelp_topics/");
//		t.lda_topics(conf, "/user/kyle/yelp_topics/part-m-00000", "/user/kyle/yelp_vec/dictionary.file-0","/user/kyle/yelp_output/");
		System.out.println("done");
//		t.readDictionaryAndFrequency("/home/kyle/yelp_vec/dictionary.file-0", "/home/kyle/yelp_vec/frequency.file-0");
	}
}

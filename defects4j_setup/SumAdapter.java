import java.util.ArrayList;
import java.util.List;

/**
 * Toy stand-in for a real per-bug adapter, used only to exercise the
 * pipeline end-to-end before wiring in an actual Defects4J checkout.
 *
 * Input format (deliberately NOT full JSON, to avoid needing any external
 * library while compiling this toy example): a comma-separated list of
 * integers, e.g. "1,2,3,4". A real adapter would instead parse whatever
 * JSON the Python side sends using a library already on the checkout's
 * classpath.
 *
 * "Buggy" behavior: silently drops any negative number instead of summing
 * it in -- a semantic bug (wrong answer, no exception), exactly the kind
 * a crash-only oracle would miss and a differential oracle catches.
 */
public class SumAdapterBuggy implements BugAdapter {
    @Override
    public String invoke(String csvInput) throws Exception {
        List<Integer> values = parse(csvInput);
        int sum = 0;
        for (int v : values) {
            if (v >= 0) {  // <-- the bug: negatives are silently skipped
                sum += v;
            }
        }
        return String.valueOf(sum);
    }

    static List<Integer> parse(String csvInput) {
        List<Integer> values = new ArrayList<>();
        if (csvInput == null || csvInput.isEmpty()) {
            return values;
        }
        for (String part : csvInput.split(",")) {
            values.add(Integer.parseInt(part.trim()));
        }
        return values;
    }
}
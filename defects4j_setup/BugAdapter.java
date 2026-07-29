/**
 * The only piece of Java you write per Defects4J bug.
 *
 * Implement invoke() to:
 *   1. deserialize jsonInput into whatever arguments the buggy method needs
 *      (use whatever JSON/parsing library is already on that project's own
 *      classpath -- Harness.java has no opinion about this and needs none),
 *   2. call the actual method under test from that Defects4J checkout,
 *   3. return a String representation of its output.
 *
 * If the method throws, let the exception propagate -- Harness.java catches
 * it and reports it to the Python side as an exception outcome. Do NOT
 * catch-and-swallow exceptions here unless the exception itself is part of
 * the expected (non-buggy) behavior you want to compare.
 *
 * Compile and keep ONE copy of your adapter class per checkout (buggy and
 * fixed), since each is compiled against that checkout's own classes.
 */
public interface BugAdapter {
    String invoke(String jsonInput) throws Exception;
}
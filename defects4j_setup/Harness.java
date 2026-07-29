/**
 * Generic persistent-JVM harness for driving one Defects4J checkout (buggy
 * OR fixed -- run one Harness process per checkout) from Python over a
 * local TCP socket.
 *
 * Why persistent: HDD/Wddmin runs dozens to hundreds of tests per bug.
 * Spawning a fresh `java` process per test (JVM startup ~100-500ms+) would
 * dominate the whole run. This harness starts once, loads your per-bug
 * BugAdapter via reflection, and then answers one request per socket
 * connection for the rest of the run.
 *
 * Wire protocol (little effort, no JSON library needed on this side --
 * BugAdapter implementations decide how to parse the payload they receive,
 * using whatever's already on the checkout's own classpath):
 *
 *   Request:  4-byte big-endian length, then that many UTF-8 bytes
 *             (the serialized candidate input -- typically JSON built by
 *             the Python side with the stdlib json module).
 *   Response: 1 status byte (0 = OK, 1 = exception was thrown),
 *             4-byte big-endian length, then that many UTF-8 bytes
 *             (the adapter's returned string, or "<ExceptionClass>: <msg>"
 *             on the exception path).
 *
 * On startup this process prints a single line "READY" to stdout once the
 * server socket is listening -- the Python side waits for that line before
 * sending any requests.
 *
 * Usage:
 *   javac -cp <buggy_checkout_classes>:. Harness.java BugAdapter.java
 *   java  -cp <buggy_checkout_classes>:. Harness <FullyQualifiedAdapterClass> <port>
 */

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

public class Harness {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: Harness <FullyQualifiedBugAdapterClass> <port>");
            System.exit(2);
        }

        String adapterClassName = args[0];
        int port = Integer.parseInt(args[1]);

        BugAdapter adapter = (BugAdapter) Class.forName(adapterClassName)
                .getDeclaredConstructor()
                .newInstance();

        try (ServerSocket server = new ServerSocket(port, 0, InetAddress.getByName("127.0.0.1"))) {
            // Signal readiness AFTER the adapter is constructed and the
            // socket is bound, so the Python side never races a request
            // against a server that isn't listening yet.
            System.out.println("READY");
            System.out.flush();

            while (true) {
                try (Socket client = server.accept()) {
                    handleOneRequest(client, adapter);
                } catch (IOException e) {
                    // A single bad connection should never take the server
                    // down -- log and keep serving.
                    System.err.println("Harness: connection error: " + e);
                }
            }
        }
    }

    private static void handleOneRequest(Socket client, BugAdapter adapter) throws IOException {
        DataInputStream in = new DataInputStream(client.getInputStream());
        DataOutputStream out = new DataOutputStream(client.getOutputStream());

        int len = in.readInt();
        byte[] requestBytes = new byte[len];
        in.readFully(requestBytes);
        String requestPayload = new String(requestBytes, StandardCharsets.UTF_8);

        byte status;
        String responsePayload;
        try {
            responsePayload = String.valueOf(adapter.invoke(requestPayload));
            status = 0;
        } catch (Throwable t) {
            responsePayload = t.getClass().getName() + ": " + t.getMessage();
            status = 1;
        }

        byte[] responseBytes = responsePayload.getBytes(StandardCharsets.UTF_8);
        out.writeByte(status);
        out.writeInt(responseBytes.length);
        out.write(responseBytes);
        out.flush();
    }
}
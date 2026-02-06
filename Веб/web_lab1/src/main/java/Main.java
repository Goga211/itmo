import com.fastcgi.FCGIInterface;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

public class Main {
    public static void main(String[] args) {
        FCGIInterface fcgi = new FCGIInterface();
        long startTime = System.nanoTime();
        while (fcgi.FCGIaccept() >= 0) {
            try {
                String requestMethod = System.getProperties().getProperty("REQUEST_METHOD");
                if (!"GET".equals(requestMethod))
                    throw new Exception("Invalid request method");

                String URI = System.getProperties().getProperty("REQUEST_URI");
                if (!URI.matches(".*/fcgi-bin/Web1.jar(\\?.*)?")) {
                    sendErrorResponse404("Invalid URL. The URL must match the expected format.");
                    continue;
                }
            } catch (Exception e) {
                sendErrorResponse405("GET");
                continue;
            }

            Dto dto = new Dto();
            try {
                dto.setVal();
                InputValidator inputValidator = new InputValidator();
                if (!inputValidator.validateInput(dto.getX(), dto.getY(), dto.getR())) {
                    sendErrorResponse400("Invalid input values. x must be in range [-3, 5], y must be in range [-3, 5], and R must be in range [1, 3].");
                    continue;
                }
            } catch (IllegalArgumentException e) {
                sendErrorResponse400(e.getMessage());
                continue;
            } catch (Exception e) {
                sendErrorResponse400("Unexpected error: " + e.getMessage());
                continue;
            }

            boolean isPointInsideCheck = CheckPointPos.isPointInside(dto.getX(), dto.getY(), dto.getR());
            String result = isPointInsideCheck ? "True" : "False";
            long Time = (System.nanoTime() - startTime) / 1000000L;

            RequestData requestData = new RequestData(
                    dto.getX(),
                    dto.getY(),
                    dto.getR(),
                    result,
                    getCurrentTime(),
                    Time
            );

            sendJsonResponse(requestData);
        }
    }

    private static void sendJsonResponse(RequestData requestData) {
        System.out.println("Content-type: application/json\n\n");
        String jsonResponse = requestData.toJson();
        System.out.println(jsonResponse);
    }

    private static void sendErrorResponse400(String errorMessage) {
        System.out.print("Status: 400 Bad Request\n");
        System.out.print("Content-type: application/json\n\n");
        String jsonResponse = String.format("{\"error\": \"%s\"}", errorMessage);
        System.out.println(jsonResponse);
    }

    private static void sendErrorResponse405(String errorMessage) {
        System.out.print("Status: 405 Method not Allowed\n");
        System.out.print("Content-type: application/json\n\n");
        String jsonResponse = String.format("{\"error\": \"%s\"}", errorMessage);
        System.out.println(jsonResponse);
    }

    private static void sendErrorResponse404(String errorMessage) {
        System.out.print("Status: 404 Not found\n");
        System.out.print("Content-type: application/json\n\n");
        String jsonResponse = String.format("{\"error\": \"%s\"}", errorMessage);
        System.out.println(jsonResponse);
    }

    private static String getCurrentTime() {
        return LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
    }

    private static final class RequestData implements Serializable {
        private final double x;
        private final double y;
        private final double r;
        private final String result;
        private final String currentTime;
        private final long executionTime;

        private RequestData(double x, double y, double r, String result, String currentTime, long executionTime) {
            this.x = x;
            this.y = y;
            this.r = r;
            this.result = result;
            this.currentTime = currentTime;
            this.executionTime = executionTime;
        }

        public double x() { return this.x; }
        public double y() { return this.y; }
        public double r() { return this.r; }
        public String result() { return this.result; }
        public String currentTime() { return this.currentTime; }
        public long executionTime() { return this.executionTime; }

        public String toJson() {
            return String.format(
                    Locale.US,
                    "{\"x\": %.2f, \"y\": %.2f, \"r\": %.2f, \"result\": \"%s\", \"current_time\": \"%s\", \"execution_time\": \"%d ms\"}",
                    this.x, this.y, this.r, this.result, this.currentTime, this.executionTime
            );
        }
    }
}

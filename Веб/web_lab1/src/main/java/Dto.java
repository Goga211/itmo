import com.fastcgi.FCGIInterface;
import java.util.HashMap;

public class Dto {
    private double x;

    private double y;

    private double r;

    public double getX() {
        return this.x;
    }

    public double getY() {
        return this.y;
    }

    public double getR() {
        return this.r;
    }

    public void setX(double x) {
        this.x = x;
    }

    public void setY(double y) {
        this.y = y;
    }

    public void setR(double r) {
        this.r = r;
    }

    public void setAll(double x, double y, double r) {
        this.x = x;
        this.y = y;
        this.r = r;
    }

    public void setVal() {
        HashMap<String, String> params = parseParams();
        String xStr = params.get("x");
        String yStr = params.get("y");
        String rStr = params.get("r");
        if (xStr == null || yStr == null || rStr == null)
            throw new IllegalArgumentException("All parameters (x, y, r) must be provided");
        try {
            this.x = Double.parseDouble(xStr);
            this.y = Double.parseDouble(yStr);
            this.r = Double.parseDouble(rStr);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Parameters x, y, and r must be valid numbers", e);
        }
    }

    private HashMap<String, String> parseParams() {
        HashMap<String, String> params = new HashMap<>();
        String queryString = FCGIInterface.request.params.getProperty("QUERY_STRING");
        if (queryString != null && !queryString.isEmpty())
            for (String pair : queryString.split("&")) {
                String[] keyValue = pair.split("=");
                if (keyValue.length > 1) {
                    params.put(keyValue[0], keyValue[1]);
                } else {
                    params.put(keyValue[0], "");
                }
            }
        return params;
    }
}

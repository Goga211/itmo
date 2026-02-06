public class InputValidator {
    public boolean validateInput(double x, double y, double r) {
        return (validateX(x) && validateY(y) && validateR(r));
    }

    private boolean validateX(double x) {
        return (x >= -3.0D && x <= 5.0D);
    }

    private boolean validateY(double y) {
        return (y >= -3.0D && y <= 5.0D);
    }

    private boolean validateR(double r) {
        return (r >= 1.0D && r <= 3.0D);
    }
}

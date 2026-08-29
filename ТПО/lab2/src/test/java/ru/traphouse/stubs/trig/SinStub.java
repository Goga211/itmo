package ru.traphouse.stubs.trig;

import ru.traphouse.functions.trig.Sin;

import java.util.HashMap;
import java.util.Map;

public class SinStub extends Sin {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(0.0,           0.0);
        TABLE.put(Math.PI / 6,   0.5);
        TABLE.put(Math.PI / 4,   Math.sqrt(2) / 2);
        TABLE.put(Math.PI / 3,   Math.sqrt(3) / 2);
        TABLE.put(Math.PI / 2,   1.0);
        TABLE.put(Math.PI,       0.0);
        TABLE.put(-Math.PI / 2, -1.0);
        TABLE.put(-Math.PI / 4, -Math.sqrt(2) / 2);
        TABLE.put(-1.0,         -0.8414709848078965);
        TABLE.put(-2.0,         -0.9092974268256817);
        TABLE.put(-3.0,         -0.1411200080598672);
    }

    public SinStub() {
        super(1e-10);
    }

    @Override
    public double calculate(double x) {
        for (Map.Entry<Double, Double> entry : TABLE.entrySet()) {
            if (Math.abs(entry.getKey() - x) < 1e-9) {
                return entry.getValue();
            }
        }
        return super.calculate(x);
    }
}

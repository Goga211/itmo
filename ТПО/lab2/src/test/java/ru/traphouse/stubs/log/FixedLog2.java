package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Log2;

/**
 * Заглушка Log2 с фиксированным возвращаемым значением.
 */
public class FixedLog2 extends Log2 {
    private final double fixedValue;

    public FixedLog2(double fixedValue) {
        super(new FixedLn(1.0));
        this.fixedValue = fixedValue;
    }

    @Override
    public double calculate(double x) {
        return fixedValue;
    }
}

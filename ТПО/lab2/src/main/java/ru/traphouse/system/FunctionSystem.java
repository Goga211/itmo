package ru.traphouse.system;

/**
 * Система функций:
 *   x <= 0 → TrigSystem
 *   x > 0  → LogSystem
 */
public class FunctionSystem {
    private final TrigSystem trigSystem;
    private final LogSystem logSystem;

    public FunctionSystem(TrigSystem trigSystem, LogSystem logSystem) {
        this.trigSystem = trigSystem;
        this.logSystem = logSystem;
    }

    public double calculate(double x) {
        if (x <= 0) {
            return trigSystem.calculate(x);
        } else {
            return logSystem.calculate(x);
        }
    }
}

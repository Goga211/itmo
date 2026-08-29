package ru.traphouse.system;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import ru.traphouse.functions.log.*;
import ru.traphouse.stubs.log.*;

import static org.junit.jupiter.api.Assertions.*;

class LogSystemTest {
    private static final double EPS = 1e-4;
    private LogSystem logSystem;
    private LogSystem logSystemWithStubs;

    @BeforeEach
    void setUp() {
        Ln    ln    = new Ln(1e-10);
        Log2  log2  = new Log2(ln);
        Log3  log3  = new Log3(ln);
        Log5  log5  = new Log5(ln);
        Log10 log10 = new Log10(ln);
        logSystem = new LogSystem(ln, log2, log3, log5, log10);

        LnStub    lnStub    = new LnStub();
        Log2Stub  log2Stub  = new Log2Stub(lnStub);
        Log3Stub  log3Stub  = new Log3Stub(lnStub);
        Log5Stub  log5Stub  = new Log5Stub(lnStub);
        Log10Stub log10Stub = new Log10Stub(lnStub);
        logSystemWithStubs = new LogSystem(lnStub, log2Stub, log3Stub, log5Stub, log10Stub);
    }

    @Test
    void testUndefinedAtOne() {
        assertThrows(ArithmeticException.class, () -> logSystem.calculate(1.0));
    }

    @Test
    void testAtTwo() {
        double x = 2.0;
        double log10v = Math.log10(x), lnv = Math.log(x);
        double log3v  = Math.log(x) / Math.log(3);
        double log5v  = Math.log(x) / Math.log(5);
        double log2v  = Math.log(x) / Math.log(2);
        double num  = ((log10v + lnv + log10v) - Math.pow(log3v, 3)) * log10v;
        double dTop = log3v + (log5v - log2v) * log2v;
        double expected = num / (dTop / -lnv);
        assertEquals(expected, logSystem.calculate(x), EPS);
    }

    @Test
    void testAtTen() {
        double x = 10.0;
        double log10v = 1.0, lnv = Math.log(x);
        double log3v  = Math.log(x) / Math.log(3);
        double log5v  = Math.log(x) / Math.log(5);
        double log2v  = Math.log(x) / Math.log(2);
        double num  = ((log10v + lnv + log10v) - Math.pow(log3v, 3)) * log10v;
        double dTop = log3v + (log5v - log2v) * log2v;
        double expected = num / (dTop / -lnv);
        assertEquals(expected, logSystem.calculate(x), EPS);
    }

    @Test
    void testStubMatchesReal() {
        assertEquals(logSystem.calculate(2.0), logSystemWithStubs.calculate(2.0), EPS);
    }

    @Test
    void testPositiveDomain() {
        assertDoesNotThrow(() -> logSystem.calculate(5.0));
        assertDoesNotThrow(() -> logSystem.calculate(0.5));
    }

}

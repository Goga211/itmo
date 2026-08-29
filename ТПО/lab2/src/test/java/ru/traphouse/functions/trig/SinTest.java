package ru.traphouse.functions.trig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import ru.traphouse.stubs.trig.SinStub;

import static org.junit.jupiter.api.Assertions.*;

class SinTest {
    private static final double EPS = 1e-6;
    private Sin sin;

    @BeforeEach
    void setUp() {
        sin = new Sin(1e-10);
    }

    @Test
    void testZero() {
        assertEquals(0.0, sin.calculate(0.0), EPS);
    }

    @Test
    void testPiHalf() {
        assertEquals(1.0, sin.calculate(Math.PI / 2), EPS);
    }

    @Test
    void testPi() {
        assertEquals(0.0, sin.calculate(Math.PI), EPS);
    }

    @Test
    void testNegative() {
        assertEquals(-1.0, sin.calculate(-Math.PI / 2), EPS);
    }

    @Test
    void testTwoPi() {
        // Проверка нормализации: sin(2π) = 0
        assertEquals(0.0, sin.calculate(2 * Math.PI), EPS);
    }

    @Test
    void testNormalizationNegativeBeyondMinusPi() {
        // x = -3π/2: после x % 2π = -3π/2 < -π → срабатывает x += 2π → нормализуется до π/2
        // sin(-3π/2) = sin(π/2) = 1
        assertEquals(1.0, sin.calculate(-3 * Math.PI / 2), EPS);
    }

    @Test
    void testStubMatchesReal() {
        SinStub stub = new SinStub();
        assertEquals(stub.calculate(-1.0), sin.calculate(-1.0), EPS);
        assertEquals(stub.calculate(-2.0), sin.calculate(-2.0), EPS);
    }
}

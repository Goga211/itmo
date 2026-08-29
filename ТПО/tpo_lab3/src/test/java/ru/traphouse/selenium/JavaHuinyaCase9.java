package ru.traphouse.selenium;

import java.time.Duration;

import io.github.cdimascio.dotenv.Dotenv;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import static org.junit.jupiter.api.Assertions.assertNotNull;

public class JavaHuinyaCase9 {
    private WebDriver driver;
    JavascriptExecutor js;

    @BeforeEach
    public void setUp() {
        driver = new FirefoxDriver();
        js = (JavascriptExecutor) driver;
    }

    @AfterEach
    public void tearDown() {
        driver.quit();
    }

    @Test
    public void authorizationTest() {
        Dotenv dotenv = Dotenv.load();

        String email = dotenv.get("TEST_EMAIL");
        String password = dotenv.get("TEST_PASSWORD");

        driver.get("https://stackoverflow.com/questions");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        // ждем кнопку Log in и кликаем
        wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("(//a[contains(text(),\'Log in\')])[2]")
        )).click();

        // ждем поле email
        wait.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//input[@id=\'email\']")));

        assertNotNull(email);
        assertNotNull(password);

        driver.findElement(By.xpath("//input[@id=\'email\']")).sendKeys(email);
        driver.findElement(By.xpath("//input[@id=\'password\']")).sendKeys(password);
        driver.findElement(By.xpath("//button[@id=\'submit-button\']")).click();

        WebElement avatar = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//a[@id=\'user-profile-button\']")
        ));

        avatar.click();
    }
}

package ru.traphouse.selenium;

import java.time.Duration;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class JavaHuinyaCase1 {
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
    public void testCase1() {
        driver.get("https://stackoverflow.com/questions");
//        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
//
//        wait.until(ExpectedConditions.elementToBeClickable(By.xpath("//span[contains(.,\'Newest\')]"))).click();
        driver.findElement(By.xpath("//span[contains(.,\'Newest\')]")).click();
        driver.findElement(By.xpath("//span[contains(.,\'Active\')]")).click();
        driver.findElement(By.xpath("//span[contains(.,\'Bountied\')]")).click();
        driver.findElement(By.xpath("//a[contains(.,\'Unanswered\')]")).click();
        driver.findElement(By.xpath("//button[contains(.,\'More\')]")).click();
        driver.findElement(By.xpath("//a[contains(.,\'Frequent\')]")).click();
        driver.findElement(By.xpath("//span[contains(.,\'More\')]")).click();
        driver.findElement(By.xpath("//a[contains(.,\'Score\')]")).click();
        driver.findElement(By.xpath("//span[contains(.,\'More\')]")).click();
        driver.findElement(By.xpath("//a[contains(.,\'Trending\')]")).click();
        driver.findElement(By.xpath("//span[contains(.,\'More\')]")).click();
        driver.findElement(By.xpath("//a[contains(.,\'Week\')]")).click();
        driver.findElement(By.xpath("//span[contains(.,\'More\')]")).click();
        driver.findElement(By.xpath("//a[contains(.,\'Month\')]")).click();
        driver.findElement(By.xpath("//a[contains(.,\'Ask Question\')]")).click();
    }
}


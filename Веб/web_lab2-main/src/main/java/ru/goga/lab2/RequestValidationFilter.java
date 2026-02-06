package ru.goga.lab2;

import jakarta.servlet.*;
import jakarta.servlet.annotation.WebFilter;
import jakarta.servlet.http.HttpServletResponse;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

@WebFilter("/controller-servlet") // Применяем фильтр к запросам на ControllerServlet
public class RequestValidationFilter implements Filter {

    private static final Logger logger = Logger.getLogger(RequestValidationFilter.class.getName());
    private ServletContext servletContext;

    @Override
    public void init(FilterConfig filterConfig) {
        logger.info("Инициализация фильтра RequestValidationFilter");
        this.servletContext = filterConfig.getServletContext();

        // Инициализируем список requestInfoMap, если его ещё нет в контексте
        if (servletContext.getAttribute("requestInfoList") == null) {
            servletContext.setAttribute("requestInfoList", new ArrayList<Map<String, Object>>());
        }
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {

        String formattedNow = getCurrentTimestamp(); // Получаем текущее время в виде строки
        boolean isValid = areParametersValid(request); // Проверяем валидность параметров

        // Мапа для текущего запроса
        Map<String, Object> currentRequestInfo = new HashMap<>();
        currentRequestInfo.put("timestamp", formattedNow);
        currentRequestInfo.put("isValid", isValid);

        // Получаем список всех запросов из контекста
        List<Map<String, Object>> requestInfoList = getRequestInfoListFromContext();
        requestInfoList.add(currentRequestInfo); // Добавляем данные текущего запроса в список

        logger.info("Текущее содержимое requestInfoList: " + requestInfoList);

        // Если запрос валиден, продолжаем цепочку фильтров
        if (isValid) {
            chain.doFilter(request, response);
        } else {
            // Если запрос не валиден, отправляем ошибку
            sendBadRequestResponse(response);
        }
    }


    // Метод для получения текущего времени в виде строки
    private String getCurrentTimestamp() {
        return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
    }

    // Метод для проверки валидности параметров
    private boolean areParametersValid(ServletRequest request) {
        String x = request.getParameter("x");
        String y = request.getParameter("y");
        String r = request.getParameter("r");
        boolean isValid = x != null && !x.isEmpty() && y != null && !y.isEmpty() && r != null && !r.isEmpty();

        if (!isValid) {
            logger.warning("Запрос не прошёл проверку: один или несколько параметров отсутствуют.");
        }
        return isValid;
    }

    // Метод для отправки ошибки при недопустимом запросе
    private void sendBadRequestResponse(ServletResponse response) throws IOException {
        HttpServletResponse httpResponse = (HttpServletResponse) response;
        httpResponse.sendError(HttpServletResponse.SC_BAD_REQUEST, "Ошибка: один или несколько параметров отсутствуют");
    }

    // Получение списка запросов из контекста
    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> getRequestInfoListFromContext() {
        List<Map<String, Object>> requestInfoList = (List<Map<String, Object>>) servletContext.getAttribute("requestInfoList");
        if (requestInfoList == null) {
            requestInfoList = new ArrayList<>();
            servletContext.setAttribute("requestInfoList", requestInfoList);
        }
        return requestInfoList;
    }
}



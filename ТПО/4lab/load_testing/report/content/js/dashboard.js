/*
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
var showControllersOnly = false;
var seriesFilter = "";
var filtersOnlySampleSeries = true;

/*
 * Add header in statistics table to group metrics by category
 * format
 *
 */
function summaryTableHeader(header) {
    var newRow = header.insertRow(-1);
    newRow.className = "tablesorter-no-sort";
    var cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 1;
    cell.innerHTML = "Requests";
    newRow.appendChild(cell);

    cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 3;
    cell.innerHTML = "Executions";
    newRow.appendChild(cell);

    cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 7;
    cell.innerHTML = "Response Times (ms)";
    newRow.appendChild(cell);

    cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 1;
    cell.innerHTML = "Throughput";
    newRow.appendChild(cell);

    cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 2;
    cell.innerHTML = "Network (KB/sec)";
    newRow.appendChild(cell);
}

/*
 * Populates the table identified by id parameter with the specified data and
 * format
 *
 */
function createTable(table, info, formatter, defaultSorts, seriesIndex, headerCreator) {
    var tableRef = table[0];

    // Create header and populate it with data.titles array
    var header = tableRef.createTHead();

    // Call callback is available
    if(headerCreator) {
        headerCreator(header);
    }

    var newRow = header.insertRow(-1);
    for (var index = 0; index < info.titles.length; index++) {
        var cell = document.createElement('th');
        cell.innerHTML = info.titles[index];
        newRow.appendChild(cell);
    }

    var tBody;

    // Create overall body if defined
    if(info.overall){
        tBody = document.createElement('tbody');
        tBody.className = "tablesorter-no-sort";
        tableRef.appendChild(tBody);
        var newRow = tBody.insertRow(-1);
        var data = info.overall.data;
        for(var index=0;index < data.length; index++){
            var cell = newRow.insertCell(-1);
            cell.innerHTML = formatter ? formatter(index, data[index]): data[index];
        }
    }

    // Create regular body
    tBody = document.createElement('tbody');
    tableRef.appendChild(tBody);

    var regexp;
    if(seriesFilter) {
        regexp = new RegExp(seriesFilter, 'i');
    }
    // Populate body with data.items array
    for(var index=0; index < info.items.length; index++){
        var item = info.items[index];
        if((!regexp || filtersOnlySampleSeries && !info.supportsControllersDiscrimination || regexp.test(item.data[seriesIndex]))
                &&
                (!showControllersOnly || !info.supportsControllersDiscrimination || item.isController)){
            if(item.data.length > 0) {
                var newRow = tBody.insertRow(-1);
                for(var col=0; col < item.data.length; col++){
                    var cell = newRow.insertCell(-1);
                    cell.innerHTML = formatter ? formatter(col, item.data[col]) : item.data[col];
                }
            }
        }
    }

    // Add support of columns sort
    table.tablesorter({sortList : defaultSorts});
}

$(document).ready(function() {

    // Customize table sorter default options
    $.extend( $.tablesorter.defaults, {
        theme: 'blue',
        cssInfoBlock: "tablesorter-no-sort",
        widthFixed: true,
        widgets: ['zebra']
    });

    var data = {"OkPercent": 33.333333333333336, "KoPercent": 66.66666666666667};
    var dataset = [
        {
            "label" : "FAIL",
            "data" : data.KoPercent,
            "color" : "#FF6347"
        },
        {
            "label" : "PASS",
            "data" : data.OkPercent,
            "color" : "#9ACD32"
        }];
    $.plot($("#flot-requests-summary"), dataset, {
        series : {
            pie : {
                show : true,
                radius : 1,
                label : {
                    show : true,
                    radius : 3 / 4,
                    formatter : function(label, series) {
                        return '<div style="font-size:8pt;text-align:center;padding:2px;color:white;">'
                            + label
                            + '<br/>'
                            + Math.round10(series.percent, -2)
                            + '%</div>';
                    },
                    background : {
                        opacity : 0.5,
                        color : '#000'
                    }
                }
            }
        },
        legend : {
            show : true
        }
    });

    // Creates APDEX table
    createTable($("#apdexTable"), {"supportsControllersDiscrimination": true, "overall": {"data": [0.16666666666666666, 500, 1500, "Total"], "isController": false}, "titles": ["Apdex", "T (Toleration threshold)", "F (Frustration threshold)", "Label"], "items": [{"data": [0.0, 500, 1500, "Config 2 "], "isController": false}, {"data": [0.5, 500, 1500, "Config 3"], "isController": false}, {"data": [0.0, 500, 1500, "Config 1"], "isController": false}]}, function(index, item){
        switch(index){
            case 0:
                item = item.toFixed(3);
                break;
            case 1:
            case 2:
                item = formatDuration(item);
                break;
        }
        return item;
    }, [[0, 0]], 3);

    // Create statistics table
    createTable($("#statisticsTable"), {"supportsControllersDiscrimination": true, "overall": {"data": ["Total", 1200, 800, 66.66666666666667, 912.6283333333329, 519, 1231, 976.5, 1191.8000000000002, 1203.0, 1224.98, 10.722806515892092, 2.4189143605186265, 1.7487389532753705], "isController": false}, "titles": ["Label", "#Samples", "FAIL", "Error %", "Average", "Min", "Max", "Median", "90th pct", "95th pct", "99th pct", "Transactions/s", "Received", "Sent"], "items": [{"data": ["Config 2 ", 400, 400, 100.0, 978.8499999999999, 913, 1099, 976.5, 1026.8000000000002, 1039.95, 1098.99, 3.579898867856983, 0.8075748422607062, 0.5838311630196447], "isController": false}, {"data": ["Config 3", 400, 0, 0.0, 586.0450000000003, 519, 682, 587.0, 619.9000000000001, 633.95, 644.97, 3.592566978920613, 0.8104325899713494, 0.585897153788811], "isController": false}, {"data": ["Config 1", 400, 400, 100.0, 1172.9899999999996, 1108, 1231, 1174.5, 1209.8000000000002, 1218.0, 1230.98, 3.5742688386306978, 0.8063047868395421, 0.5829129844251235], "isController": false}]}, function(index, item){
        switch(index){
            // Errors pct
            case 3:
                item = item.toFixed(2) + '%';
                break;
            // Mean
            case 4:
            // Mean
            case 7:
            // Median
            case 8:
            // Percentile 1
            case 9:
            // Percentile 2
            case 10:
            // Percentile 3
            case 11:
            // Throughput
            case 12:
            // Kbytes/s
            case 13:
            // Sent Kbytes/s
                item = item.toFixed(2);
                break;
        }
        return item;
    }, [[0, 0]], 0, summaryTableHeader);

    // Create error table
    createTable($("#errorsTable"), {"supportsControllersDiscrimination": false, "titles": ["Type of error", "Number of errors", "% in errors", "% in all samples"], "items": [{"data": ["The operation lasted too long: It took 1,021 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,125 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,182 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,140 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 947 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,229 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,150 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,212 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 930 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,135 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 972 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 999 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,028 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,016 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 952 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 925 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 994 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,172 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, 1.25, 0.8333333333333334], "isController": false}, {"data": ["The operation lasted too long: It took 1,162 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 957 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 962 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,202 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,157 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,199 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 989 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 959 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,113 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,194 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,033 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 992 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,200 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,120 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 949 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 960 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,001 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,004 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,222 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,177 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 982 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 940 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,174 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 969 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 927 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,184 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 950 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,187 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,142 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 929 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,098 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 986 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 990 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, 1.25, 0.8333333333333334], "isController": false}, {"data": ["The operation lasted too long: It took 933 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 923 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,024 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,205 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,118 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,185 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 996 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,025 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,138 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 970 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,190 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 928 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 997 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,165 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 976 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,186 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 980 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,175 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,014 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,180 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,196 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, 1.75, 1.1666666666666667], "isController": false}, {"data": ["The operation lasted too long: It took 1,128 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 991 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 939 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 938 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,176 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, 1.75, 1.1666666666666667], "isController": false}, {"data": ["The operation lasted too long: It took 1,214 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 942 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,131 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,005 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,141 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, 1.75, 1.1666666666666667], "isController": false}, {"data": ["The operation lasted too long: It took 1,127 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,099 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 987 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, 1.25, 0.8333333333333334], "isController": false}, {"data": ["The operation lasted too long: It took 967 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,170 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 981 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,147 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,204 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,108 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 913 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,121 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,160 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,195 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 919 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 961 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 932 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,166 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,044 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 948 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, 1.25, 0.8333333333333334], "isController": false}, {"data": ["The operation lasted too long: It took 1,137 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 977 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,208 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,017 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,146 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 983 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,188 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 993 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 958 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,156 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,161 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,201 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,168 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,213 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,022 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 978 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,193 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 941 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 936 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,032 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,027 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,183 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,223 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,218 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,178 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 998 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 956 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,158 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 995 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,029 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,148 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 966 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, 1.25, 0.8333333333333334], "isController": false}, {"data": ["The operation lasted too long: It took 963 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,225 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 946 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 943 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,007 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,126 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 988 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,091 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 985 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,206 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,039 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,203 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,171 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 924 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,020 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 953 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,003 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,164 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,139 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 954 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,143 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,153 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 975 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,013 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,066 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 955 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,210 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,019 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,231 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,009 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,149 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, 1.25, 0.8333333333333334], "isController": false}, {"data": ["The operation lasted too long: It took 1,040 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,133 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 944 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 965 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, 1.25, 0.8333333333333334], "isController": false}, {"data": ["The operation lasted too long: It took 1,154 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,134 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,173 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, 1.25, 0.8333333333333334], "isController": false}, {"data": ["The operation lasted too long: It took 1,217 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,090 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 935 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,008 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,221 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,179 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,207 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,144 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 984 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,211 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 945 milliseconds, but should not have lasted longer than 910 milliseconds.", 4, 0.5, 0.3333333333333333], "isController": false}, {"data": ["The operation lasted too long: It took 1,002 milliseconds, but should not have lasted longer than 910 milliseconds.", 6, 0.75, 0.5], "isController": false}, {"data": ["The operation lasted too long: It took 1,227 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,169 milliseconds, but should not have lasted longer than 910 milliseconds.", 8, 1.0, 0.6666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 1,163 milliseconds, but should not have lasted longer than 910 milliseconds.", 2, 0.25, 0.16666666666666666], "isController": false}, {"data": ["The operation lasted too long: It took 974 milliseconds, but should not have lasted longer than 910 milliseconds.", 12, 1.5, 1.0], "isController": false}, {"data": ["The operation lasted too long: It took 1,192 milliseconds, but should not have lasted longer than 910 milliseconds.", 12, 1.5, 1.0], "isController": false}]}, function(index, item){
        switch(index){
            case 2:
            case 3:
                item = item.toFixed(2) + '%';
                break;
        }
        return item;
    }, [[1, 1]]);

        // Create top5 errors by sampler
    createTable($("#top5ErrorsBySamplerTable"), {"supportsControllersDiscrimination": false, "overall": {"data": ["Total", 1200, 800, "The operation lasted too long: It took 1,196 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, "The operation lasted too long: It took 1,176 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, "The operation lasted too long: It took 1,141 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, "The operation lasted too long: It took 974 milliseconds, but should not have lasted longer than 910 milliseconds.", 12, "The operation lasted too long: It took 1,192 milliseconds, but should not have lasted longer than 910 milliseconds.", 12], "isController": false}, "titles": ["Sample", "#Samples", "#Errors", "Error", "#Errors", "Error", "#Errors", "Error", "#Errors", "Error", "#Errors", "Error", "#Errors"], "items": [{"data": ["Config 2 ", 400, 400, "The operation lasted too long: It took 974 milliseconds, but should not have lasted longer than 910 milliseconds.", 12, "The operation lasted too long: It took 990 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, "The operation lasted too long: It took 987 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, "The operation lasted too long: It took 948 milliseconds, but should not have lasted longer than 910 milliseconds.", 10, "The operation lasted too long: It took 966 milliseconds, but should not have lasted longer than 910 milliseconds.", 10], "isController": false}, {"data": [], "isController": false}, {"data": ["Config 1", 400, 400, "The operation lasted too long: It took 1,196 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, "The operation lasted too long: It took 1,176 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, "The operation lasted too long: It took 1,141 milliseconds, but should not have lasted longer than 910 milliseconds.", 14, "The operation lasted too long: It took 1,192 milliseconds, but should not have lasted longer than 910 milliseconds.", 12, "The operation lasted too long: It took 1,172 milliseconds, but should not have lasted longer than 910 milliseconds.", 10], "isController": false}]}, function(index, item){
        return item;
    }, [[0, 0]], 0);

});

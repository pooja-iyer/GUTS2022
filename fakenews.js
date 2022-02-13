chrome.tabs.getSelected(null,function(tab) {
	var tablink = tab.url;
	//alert(tablink);
});
document.addEventListener('DOMContentLoaded', function(tab)
{
	
	var xValues = ["Fake","Real"]; <!-- add data from backend-->
                var yValues = [65,35]; <!-- add data from backend-->
                var barColors = 'rgba(255, 159, 64, 0.2)';
                var barborder = 'rgb(255, 159, 64)';

                new Chart("myChart", {
                            type: "bar",
                            data: {
                            labels: xValues,
                            datasets: [{
                                backgroundColor: barColors,
                                data: yValues,
                                borderColor: barborder,
                                borderWidth: 1
                            }]
                        },
                    options: {
                        legend: {display: false},
                        title: {
                            display: true,
                            text: "Fake News Probability"
                        }
                    }
                });
});

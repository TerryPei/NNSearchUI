// click task call task() function
function task(){
    var task = document.getElementById("search").value;
    // const url = "https://dictionary.cambridge.org/dictionary/english/";
    const url = "https://huggingface.co/models?pipeline_tag=";
    // const database = 
    var url1 = url + task;
    var win = window.open(url1, '_blank');
    win.focus;
}
// click model call model() function
function model(){
    // window.location="graph.html"
    var model = document.getElementById("search").value;
    $.ajax({
        type: "POST",
        url: "{{ url_for('search_engine') }}",
        data: {"search_garph" : model},
        // contentType: "application/json",
        // dataType: 'json',
        // success: callbackFunc()
    })
}
// function callbackFunc(response) {
//     // do something with the response
//     console.log(response);
// }
function searchKeyPress(e){
    // look for window.event in case event isn't passed in
    e = e || window.event;
    if (e.keyCode == 13){
        var model = document.getElementById("search").value;
        const url = "https://huggingface.co/";
        // const database = 
        var url1 = url + model;
        var win = window.open(url1, '_blank');
        win.focus;   
    }
}
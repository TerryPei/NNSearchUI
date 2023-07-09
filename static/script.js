function Retrieval(){

    var title = document.getElementById("title");

    var model = document.getElementById("search").value;

    var server_data = [
        {"query": model},
    ];

    var cards = document.querySelectorAll(".card");

    cards.forEach((card) => {
      card.style.opacity = 0;
    });

    let delay = 0;
    cards.forEach((card) => {
      setTimeout(() => {
        card.style.opacity = 1;
      }, delay);
      delay += 100;
    });

    $.ajax({
        type: "POST",
        url: "/retrieval",
        data: JSON.stringify(server_data),
        contentType: "application/json",
        dataType: 'json',
        success: function(top_k_models){
          title.innerHTML = "<h2 style='text-align: center;'> Retrieval Similar Models: </h2>";
          Object.keys(top_k_models).forEach((model_name, i) => {
              $(cards[i]).html("");
              var fig_path = top_k_models[model_name]['fig_path'];
              // var img = $('<img id="dynamic">'); //Equivalent: $(document.createElement('img'))
              // img.attr('src', fig_path);
              var modelNameDiv = $('<div>').text(model_name);
              $(cards[i]).append(modelNameDiv);
              $(cards[i]).append(fig_path);
          });
      }
    });
}

function Extract(){
    var model = document.getElementById("search").value;

    var server_data = [
        {"query": model},
    ];

    var cards = document.querySelectorAll(".card");

    $.ajax({
        type: "POST",
        url: "/search_engine",

        data: JSON.stringify(server_data),
        contentType: "application/json",
        dataType: 'json',
        success: function(results){
          title.innerHTML = "<h2 style='text-align: center;'> Extracted Subgraphs: </h2>"
          // $(cards[i]).html("");
          for (var i = 0; i < results['fig_path'].length; i++) {
            var card = cards[i];
            $(card).append(results['model_names'][i])
            $(card).append(results['fig_path'][i]);
          }
        }
    });
}

function searchKeyPress(e){
    e = e || window.event;
    if (e.keyCode == 13){
        var task = document.getElementById("search").value;
        const url = "https://huggingface.co/models?pipeline_tag=";
        var url1 = url + task;
        var win = window.open(url1, '_blank');
        win.focus;
    }
}

const fileUpload = document.getElementById("file-upload");
const uploadedImage = document.getElementById("uploaded-image");

fileUpload.addEventListener("change", () => {

  const file = fileUpload.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = () => {
      uploadedImage.src = reader.result;
      uploadedImage.classList.add("show");
    };
    reader.readAsDataURL(file);
  } else {
    uploadedImage.src = "";
    uploadedImage.classList.remove("show");
  }
  var model = file ? file.name.slice(0, -4) : "";

  var server_data = [
      {"query": model},
  ];

  var cards = document.querySelectorAll(".card");

  cards.forEach((card) => {
    card.style.opacity = 0;
  });

  let delay = 0;
  cards.forEach((card) => {
    setTimeout(() => {
      card.style.opacity = 1;
    }, delay);
    delay += 100;
  });

  $.ajax({
      type: "POST",
      url: "/retrieval",
      data: JSON.stringify(server_data),
      contentType: "application/json",
      dataType: 'json',
      success: function(results){
        title.innerHTML = "<h2 style='text-align: center;'> Retrieval Similar Models: </h2>";
        
        for (var i = 0; i < results['model_names'].length; i++) {
          var card = cards[i];
          $(card).append(results['model_names'][i]);
          $(card).append(results['similar_scores'][i]);
          $(card).append(results['fig_path'][i]);
        } 
      }
  });
});
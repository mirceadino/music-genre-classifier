var host = "http://127.0.0.1:5000/predict";

function load() {
  $("#loader").show();
  $("#error").hide();
  $("#results").hide();
}

function display_genre(data) {
  var music_genre = data["predicted_genre"];
  var counting = data["counting"];
  var total = data["total"];

  if(music_genre != "None") {
    $("#music_genre").text("It's most likely: " + music_genre + "!");
  } else {
    $("#music_genre").text("We can't predict it with confidence.");
  }

  $("#counting").empty();
  Object.keys(counting).forEach(function(key) {
    var bar = document.createElement("div");
    bar.className = "progress-bar progress-bar-danger";
    bar.setAttribute("role", "progressbar");
    bar.setAttribute("aria-valuemin", 0);
    bar.setAttribute("aria-valuemax", total);
    bar.setAttribute("aria-valuenow", counting[key]);
    bar.style.width = counting[key]/total*100 + "%";

    var genre = document.createElement("span");
    genre.className = "h4";
    genre.style.margin= 0;
    genre.textContent = key;

    var bar_container = document.createElement("div");
    bar_container.className = "progress";
    bar_container.appendChild(bar);
    bar_container.appendChild(genre);
    if(key != music_genre && music_genre != "None") {
      bar_container.style.opacity = 0.5;
    }

    $("#counting").append(bar_container);
  });
}

function display_error(error) {
  $("#loader").hide();
  $("#error").show();
  $("#results").hide();

  $("#error").text(error);
}

$(document).ready(function() {
  $("#submit").click(function() {
    load();

    var url = $("#url").val();
    var data = {url: url};

    $.ajax({
      type: "POST",
      url: host,
      data: JSON.stringify(data),
      dataType: "text",
      contentType: "application/json; charset=utf-8",
      success: function (data, status, xhr) {
        $("#loader").hide();
        $("#error").hide();
        $("#results").show();

        display_genre($.parseJSON(data));
      },
      error: function (data, status, xhr) {
        display_error("We could not classify the song. Check if the URL is valid.");
      }
    });
  });
})

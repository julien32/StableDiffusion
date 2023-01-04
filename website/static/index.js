function deleteNote(noteId) {
  fetch("/delete-note", {
    method: "POST",
    body: JSON.stringify({ noteId: noteId }),
  }).then((_res) => {
    window.location.href = "/";
  });
}

// function run_text2Img(prompt) {
//   fetch('/run_text2Img')
//     .then(response => response.text())
//     .then(data => {
//       console.log(data);
//     });
// }
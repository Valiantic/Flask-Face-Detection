document.getElementById("upload-form").onsubmit = async function(event) {
    event.preventDefault();
    let formData = new FormData(this);
    let response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    let result = await response.json();
    let imgUrl = URL.createObjectURL( 
        await fetch(result.output_path).then((res) => res.blob())
    );

    document.getElementById('result').innerHTML = `
        <p>Number of faces detected: <strong>${result.num_faces}</strong></p>
        <img width="500px" src="${imgUrl}" alt="Result" />
    `;
};
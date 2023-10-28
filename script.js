const image_input = document.querySelector("#file");
var upload_image = ""; //hyper varible

image_input.addEventListener("change", function(){
    const reader = new FileReader();
    reader.addEventListener("load", () => {
        upload_image = reader.result;
        document.querySelector("#display_image").style.backgroundImage = `url(${upload_image})`;
    });
    reader.readAsDataURL(this.files[0]);
})
const imageInput = document.getElementById("image-input");
const uploadBtn = document.getElementById("upload-btn");

uploadBtn.addEventListener("click", async () => {
  const files = imageInput.files;
  if (files.length === 0 || files.length > 20) {
      alert("Vui lòng chọn từ 1 đến 20 ảnh");
      return;
  }
  const loadingBar = document.querySelector('.loading-bar-wrapper');
  loadingBar.style.display = 'block';

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
      formData.append("image", files[i]);
  }

  try {
      const response = await fetch("/segment", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      sessionStorage.setItem('original_images', JSON.stringify(data.original_images));
      sessionStorage.setItem('segmented_image', data.segmented_image);

      window.location.href = `/result?predicted_area=${data.area}&execution_time=${data.execution_time}`;
  } catch (error) {
      console.error("Error:", error);
      alert("Đã xảy ra lỗi. Vui lòng thử lại.");
  } finally {
      loadingBar.style.display = 'none';
  }
});

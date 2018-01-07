$(document).ready(() => {
	
	$("#file-upload").change(() => {

		var input = $('#file-upload')[0];
		if (input.files && input.files[0]) {
			var reader = new FileReader();

			reader.onload = function(e) {
			  	$('#image-display').attr('src', e.target.result);
				$('#image-display').show();	
			}

			reader.readAsDataURL(input.files[0]);

			var formData = new FormData();
			formData.append("image", input.files[0]);

			$.ajax({
				url : '/upload',
				type: 'POST',
				data: formData,
				processData: false,
				contentType: false,
				success: function(data) {
					$('#prediction').text($('#prediction').text() + data);
				}
			});
		}
	});

});

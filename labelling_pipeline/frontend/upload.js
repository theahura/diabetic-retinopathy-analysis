$(document).ready(() => {

	mdc.autoInit();

	function submitForm(input) {

		var formData = new FormData();
		formData.append("image", input.files[0]);

		$('input[type=checkbox], input[type=radio]').each(function() {
			var key = $(this).attr('id');
			var value = $(this).is(':checked');
			$(this).prop('checked', false);

			formData.append(key, value);
		});

		$('#severity-1').prop('checked', true);
		$('.submit-button').prop('disabled', true);
		$('#upload-image-display').hide();

		console.log(formData);
		console.log(input.files[0]);
		console.log('sending request');
		
		$.ajax({
			url : '/upload',
			type: 'POST',
			data: formData,
			processData: false,
			contentType: false,
			success: function(data) {
				console.log('success');
			},
			error: function(jqXHR, textStatus, errorThrown) {
				console.log(jqXHR);
				console.log(textStatus);
				console.log(errorThrown);
				var err = 'Request failed: ' + errorThrown + '. Please contact morningsidelabs@gmail.com.';
				alert(err);
			}
		});
	}


	var input = $('#file-upload')[0];
	$("#file-upload").change(() => {

		if (input.files && input.files[0]) {
			var reader = new FileReader();
			reader.onload = function(e) {
			  	$('#upload-image-display').attr('src', e.target.result);
				$('#upload-image-display').show();	
				$('.submit-button').prop('disabled', false);
			}
			reader.readAsDataURL(input.files[0]);
		}
	});

	$('.submit-button').click(() => {
		submitForm(input);
	});
});

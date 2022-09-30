window.onload = () => {
	$('#sendbutton').click(() => {
		imagebox1 = $('#imagebox1')
		input = $('#imageinput')[0]
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('image' , input.files[0])
			console.log(input.files[0])
			$.ajax({
				url: "http://192.168.1.11:5000/detectObject", // fix this to your liking
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data1){
					console.log(data1);
					console.log(data1.data.time);
					img_base64 = data1['data']['img_base64']
					string_bankNumber = data1['data']['string_bankNumber']
					string_bankName = data1['data']['string_bankName']
					string_userName = data1['data']['string_userName']
					times = data1['data']['time']
					image = img_base64.split('\'')[1]
					imagebox.attr('src' , 'data:image/jpeg;base64,'+image)
					document.getElementById("bank_number").value = string_bankNumber
					document.getElementById("bank_name").value = string_bankName
					document.getElementById("user_name").value = string_userName
					document.getElementById("times").value = times
				}
			});
		}
	});
};



function readUrl(input){
	imagebox = $('#imagebox')
	console.log("evoked readUrl")
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){
			// console.log(e)
			
			imagebox.attr('src',e.target.result); 
			imagebox.height(360);
			imagebox.width(480);
		}
		reader.readAsDataURL(input.files[0]);
	}
	console.log(input.files)
	
}
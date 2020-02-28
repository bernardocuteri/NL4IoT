package it.unical.nl4iot.nl4iotframework.services;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import it.unical.nl4iot.nl4iotframework.model.NlApplication;

@Service
public class NlAppJsonConsumer {
	
	public NlApplication getNlApplication(String nlAwareHttpServiceUrl, String httpMethod) {
		
		 RestTemplate restTemplate = new RestTemplate();
		 if(httpMethod.equalsIgnoreCase("get"))
			 return restTemplate.getForEntity(nlAwareHttpServiceUrl, NlApplication.class).getBody();
		 return restTemplate.postForEntity(nlAwareHttpServiceUrl, null, NlApplication.class).getBody();
		
	}

}

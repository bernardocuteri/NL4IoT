package it.unical.nl4iot.nl4iotframework.model.intents;

import java.util.Map;

public class Intent {

	private String name;

	private Map<String, String> slots;

	public Intent() {
		super();
	}

	public Intent(String name, Map<String, String> slots) {
		super();
		this.name = name;
		this.slots = slots;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public Map<String, String> getSlots() {
		return slots;
	}

	public void setSlots(Map<String, String> slots) {
		this.slots = slots;
	}

}

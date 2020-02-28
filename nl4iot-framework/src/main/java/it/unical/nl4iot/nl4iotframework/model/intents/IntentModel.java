package it.unical.nl4iot.nl4iotframework.model.intents;

import java.util.List;

public class IntentModel {

	private String name;

	private List<String> slots;

	private List<String> taggedExamples;

	public IntentModel() {
		super();
	}

	public IntentModel(String name, List<String> slots, List<String> taggedExamples) {
		super();
		this.name = name;
		this.slots = slots;
		this.taggedExamples = taggedExamples;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public List<String> getSlots() {
		return slots;
	}

	public void setSlots(List<String> slots) {
		this.slots = slots;
	}

	public List<String> getTaggedExamples() {
		return taggedExamples;
	}

	public void setTaggedExamples(List<String> taggedExamples) {
		this.taggedExamples = taggedExamples;
	}

}

import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { SalesForecastService } from './sales-forecast.service';

@Component({
  selector: 'app-sales-forecast',
  templateUrl: './sales-forecast.component.html',
  styleUrls: ['./sales-forecast.component.css']
})
export class SalesForecastComponent implements OnInit {

  form!: FormGroup;

  constructor(private forecastService: SalesForecastService, private fb: FormBuilder) { }


  ngOnInit(): void {
      this.form = this.fb.group({
        title:[''],
        body:['']
      })
  }

  onSubmit(){
    this.forecastService.upload(
      this.form.value
    ).subscribe()
  }

  routeToMl(){
    window.location.href = "http://127.0.0.1:5000/plot.png";
  }
}
